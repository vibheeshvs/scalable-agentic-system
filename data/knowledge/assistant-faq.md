# Assistant FAQ

## What can the assistant do?

It can call any API in its tool catalog on your behalf (invoices, payments, refunds, disputes, payouts,
subscriptions, reporting, webhooks) and answer questions from this knowledge base. Ask "what tools do you
have for X?" to see the relevant capabilities.

## Will it do things without asking me?

Reading data never needs confirmation. Anything that moves money, notifies a customer or deletes data
(sending invoices, refunds, payouts, cancellations, accepting dispute claims) always shows you the exact API
request first and waits for your approval. Creating drafts does not need approval.

## What happens if something fails?

Temporary errors (timeouts, rate limits, provider outages) are retried automatically with backoff. If the
provider rejects a request because an argument is wrong, the assistant corrects it and tries again, up to
two times, and then tells you what went wrong. Writes carry an idempotency key, so a retry never sends an
invoice or a payment twice.

## Can I check on an earlier request?

Yes. Ask "what's the status of my last request?" and the assistant will look up its own activity log,
including every API call it made and whether it succeeded.
