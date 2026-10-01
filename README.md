# Scalable agentic system: an agent for 100s to 1,000s of API tools

A chat agent that operates the full PayPal REST API (112 operations, ingested from PayPal's official OpenAPI specs) and stays accurate as the catalog grows. It's tested up to 1,095 tools across PayPal, Stripe, Slack and Twilio. It also has a RAG tool over a knowledge base and a system-search tool that can answer questions about its own capabilities and its request history.

**The design write-up, which answers each part of the task, is in [DESIGN.md](DESIGN.md).**

The core idea: the model never sees the whole catalog. A planner breaks the request into steps, a hybrid retriever picks ~8 candidate tools per step (scoped to the user's connected services), the LLM chooses among those, and plain code handles validation, human confirmation for risky actions, auth, retries, idempotency and pagination.

```
you> Send an invoice for $50 to vibheesh@example.com

----------------------------------------------------------------------
  Confirmation needed (high risk): Send invoice
  Step 2: Send the invoice created in step 1
  POST /v2/invoicing/invoices/INV2-9422-20BE-7E4E-4F5C-812E/send
----------------------------------------------------------------------
  Approve? [y/N] y
```

## Quick start

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env  -> then paste your GOOGLE_API_KEY
pytest                      # 48 offline tests, no key needed
python scripts/live_check.py   # the task's example requests against the real model (mock PayPal)
python -m agent.cli         # chat with it in the terminal
python ui/server.py         # or in the browser: http://127.0.0.1:8765
```

The default model is **Gemini (`gemini-3.8-flash`) on Google AI Studio's free tier**, so it runs without paying. Get a key at [aistudio.google.com](https://aistudio.google.com). OpenAI, Anthropic and other providers work by changing `LLM_MODEL` in `.env`; nothing else changes.

One thing to know about the free tier: when I ran this, it allowed **20 requests per model per day**, and a two-step request uses 5 (router, planner, two selections, reply). `.env.example` therefore puts the router on a second model (`ROUTER_MODEL`), which has its own quota, and that is enough for one full `live_check.py` run a day. Gemini's flash models also return `503 high demand` in bursts, so `.env.example` names a fallback model (`LLM_FALLBACK_MODEL`) that is tried when the main one is busy. If every model fails, the agent says so in its reply instead of crashing; a paid key removes the daily limit.

Without PayPal credentials the agent runs against a built-in **mock PayPal** (invoices, transactions with pagination and the 31-day search limit, balances, disputes; generic responses for everything else), so it's safe to try anything. To use the real **PayPal sandbox**, set `PAYPAL_CLIENT_ID` and `PAYPAL_CLIENT_SECRET`.

Things to try:

```
Send an invoice for $50 to vibheesh@example.com
What was my total sales volume last month?
Is there a dispute open from user_123?
What tools are available for managing invoices?
What's the status of my last request?
When do we add a late fee to an invoice?
```

CLI commands: `/tools <text>` shows what retrieval returns for a query, `/log` shows recent runs and API calls, `/new` starts a fresh thread, `/quit` exits.

### Web console

`python ui/server.py` serves a small local console at http://127.0.0.1:8765 (standard library only, nothing extra to install). It has four views:

- **Chat**: the plan as it runs, with each step's kind, the tool chosen, the tools retrieved for it, the arguments and a result preview. Risky actions stop at an approval card showing the exact request.
- **Tool retrieval**: type a step goal and see the top-k tools the model would be shown, their scores, the prompt cost next to binding the whole catalog, and which schemas were shortened to fit the provider's limits.
- **Catalog**: every tool by group, with its risk level and whether the provider allows a safe retry.
- **Activity log**: each request and the tool calls it made.

The sidebar switches between the **live model** from `.env` and an **offline demo**. The demo replaces the model with a scripted stand-in that only knows the example requests, so the console can be explored without an API key or quota. Everything else in the demo (retrieval, validation, approval, the executor, the mock PayPal, the run log) is the real code, and the page says which mode it is in.

### Observability

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to trace every turn in LangSmith. Each graph node is a span, and the LLM calls are named `router` / `planner` / `selector` / `responder` / `rag_generate`. Tool retrieval appears as a retriever run with the candidate tools and their scores.

## Tests and evals

```bash
pytest                                   # 48 offline tests: graph flows with a scripted LLM + mock PayPal,
                                         # plus the real Gemini adapter with only the network call faked
python scripts/live_check.py             # needs a key: the task's example requests end to end, writes live_check_report.md

python scripts/fetch_external_specs.py   # Stripe, Slack, Twilio specs for the scale test (~11 MB, pinned commits)
python -m agent.registry.build --all     # -> data/catalog/all.json (1,095 tools)
python scripts/eval_retrieval.py         # recall@k as the catalog grows 112 -> 1,095, scoped vs unscoped
python scripts/eval_selection.py --limit 20 --pause 4   # needs a paid key (40 calls): tool-choice accuracy, all tools vs top-k
```

Latest retrieval results are in [data/eval/retrieval_results.md](data/eval/retrieval_results.md).

## Repo layout

```
agent/
  registry/        tool catalog: OpenAPI + Postman loaders, schema compaction, risk levels, policy overrides
  retrieval/       BM25 + optional embeddings (OpenAI or offline WordLlama), RRF fusion, service scoping
  graph/           LangGraph state, nodes (router, planner, select, validate, guard, execute, respond), wiring
  tools/           HTTP executor (auth, retries, idempotency, pagination), mock PayPal,
                   built-ins: rag_search, system_search, analyze_data, ask_user
  runlog.py        SQLite audit log of runs and API calls (read by system_search)
  llm.py           provider-agnostic LLM wrapper + scripted test double
  prompts.py       router / planner / selector / responder prompts and output schemas
  cli.py           terminal chat with human-in-the-loop confirmations
data/
  specs/paypal/    PayPal's official OpenAPI specs (Apache-2.0, see LICENSE there)
  catalog/         built catalogs (paypal.json is committed; all.json is generated)
  policy/          per-tool overrides (disable / change risk)
  knowledge/       sample knowledge base for the RAG tool (fictional company docs)
  eval/            retrieval eval queries and results
scripts/           evals and spec download
tests/             pytest suite
ui/                local web console (server.py + index.html)
```

## Using a Postman collection

The brief frames the PayPal APIs as a Postman collection. The catalog here is built from PayPal's OpenAPI specs because they carry real parameter schemas, but an exported Postman collection works too (schemas are inferred from example requests):

```bash
python -m agent.registry.build --postman path/to/PayPal.postman_collection.json
CATALOG_PATH=data/catalog/paypal_postman.json python -m agent.cli
```

## Configuration

Everything is set via environment variables; see [.env.example](.env.example). The main ones:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL` / `ROUTER_MODEL` | `google_genai:gemini-3.8-flash` | any `init_chat_model` string (`google_genai:...`, `openai:...`, `anthropic:...`); the router can use a lighter model |
| `EMBEDDINGS` | `auto` | `openai`, `wordllama` (offline), or `none` (BM25 only) |
| `TOOL_TOP_K` | `8` | tools retrieved per step |
| `ENABLED_SERVICES` | all in catalog | services this user has connected (retrieval scope + hard check) |
| `CATALOG_PATH` | `data/catalog/paypal.json` | use `data/catalog/all.json` for the 1,095-tool catalog |
| `CONFIRM_RISK` | `high` | risk levels that need human approval |
| `MAX_REPAIRS` | `2` | retries after a validation or API error |
| `LLM_FALLBACK_MODEL` | none | a second model tried straight away when the main one is overloaded or out of quota |
| `LLM_MAX_RETRIES` | `2` | extra rounds after a temporary model failure, with a 5 s / 15 s wait (kept low: on the free tier every attempt counts against the daily quota) |
