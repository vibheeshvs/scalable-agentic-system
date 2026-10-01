"""Build tool catalogs from the specs in data/specs.

    python -m agent.registry.build            # PayPal only  -> data/catalog/paypal.json
    python -m agent.registry.build --all      # + Stripe, Slack, Twilio -> data/catalog/all.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import Catalog, Risk
from .openapi import load_openapi
from .postman import load_postman

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "data" / "specs"
CATALOG_DIR = ROOT / "data" / "catalog"

# Sandbox base URL for everything PayPal. A couple of PayPal specs list the live host first,
# and I never want a dev agent pointed at live money by accident.
PAYPAL_BASE = "https://api-m.sandbox.paypal.com"

EXTERNAL = {  # service -> (file, title, idempotency header accepted on every write, if any)
    "stripe": ("stripe.json", "Stripe API", "Idempotency-Key"),
    "slack": ("slack.json", "Slack Web API", None),
    "twilio": ("twilio.json", "Twilio REST API", None),
}


POLICY = ROOT / "data" / "policy" / "tool_policy.json"


def apply_policy(cat: Catalog, path: Path = POLICY) -> Catalog:
    """Human overrides on top of the heuristics: fix a risk level, or disable a tool outright."""
    if not path.exists():
        return cat
    for tool_id, rule in json.loads(path.read_text(encoding="utf-8")).items():
        tool = cat.tools.get(tool_id)
        if tool is None or tool_id.startswith("_"):
            continue
        if rule == "disabled":
            del cat.tools[tool_id]
            groups = cat.services[tool.service].groups
            groups[tool.group] -= 1
            if not groups[tool.group]:
                del groups[tool.group]
        else:
            tool.risk = Risk(rule)
    return cat


def build_paypal(spec_dir: Path = SPECS / "paypal") -> Catalog:
    cat = Catalog()
    first = True
    for f in sorted(spec_dir.glob("*.json")):
        tools, svc = load_openapi(f, service="paypal", base_url=PAYPAL_BASE)
        if first:
            svc.title, svc.description = "PayPal REST APIs", "Invoicing, orders, payments, disputes, payouts, subscriptions, reporting, webhooks."
            cat.services["paypal"] = svc
            first = False
        cat.add(tools)
    return apply_policy(cat)


def build_all(spec_dir: Path = SPECS) -> Catalog:
    cat = build_paypal(spec_dir / "paypal")
    for service, (fname, title, idem_header) in EXTERNAL.items():
        f = spec_dir / "external" / fname
        if not f.exists():
            raise FileNotFoundError(f"{f} missing - run: python scripts/fetch_external_specs.py")
        tools, svc = load_openapi(f, service=service, title=title, idempotency_header=idem_header)
        cat.services[service] = svc
        cat.add(tools)
    return apply_policy(cat)


def add_postman(cat: Catalog, path: str | Path, service: str, base_url: str) -> Catalog:
    tools, svc = load_postman(path, service=service, base_url=base_url)
    cat.services.setdefault(service, svc)
    cat.add(tools)
    return apply_policy(cat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also load the external specs (scale test)")
    ap.add_argument("--postman", help="path to an exported Postman collection to ingest instead of the OpenAPI specs")
    ap.add_argument("--service", default="paypal")
    args = ap.parse_args()

    if args.postman:
        cat = add_postman(Catalog(), args.postman, args.service, PAYPAL_BASE if args.service == "paypal" else "")
        out = CATALOG_DIR / f"{args.service}_postman.json"
    elif args.all:
        cat, out = build_all(), CATALOG_DIR / "all.json"
    else:
        cat, out = build_paypal(), CATALOG_DIR / "paypal.json"
    cat.save(out)
    print(f"{len(cat)} tools across {len(cat.services)} service(s) -> {out.relative_to(ROOT)}")
    for s in cat.services.values():
        print(f"  {s.name}: {sum(s.groups.values())} tools in {len(s.groups)} groups")


if __name__ == "__main__":
    main()
