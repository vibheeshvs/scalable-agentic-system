"""Download the public OpenAPI specs used for the 1,000+ tool scale test, then build the big catalog.

    python scripts/fetch_external_specs.py && python -m agent.registry.build --all
"""

from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parents[1] / "data" / "specs" / "external"
# Pinned to exact commits: these specs change weekly, and the numbers in data/eval/retrieval_results.md
# (and the tool count of 1,095) are only reproducible against the same files.
SPECS = {
    "stripe.json": "https://raw.githubusercontent.com/stripe/openapi/bcfd59f49dcac84a84d63d304c62793dff6e5916/openapi/spec3.json",
    "slack.json": "https://raw.githubusercontent.com/slackapi/slack-api-specs/dfea73e06d146c368d7f94b52ac90796dc4e27e1/web-api/slack_web_openapi_v2.json",
    "twilio.json": "https://raw.githubusercontent.com/twilio/twilio-oai/218b7821602a93ae63e83e20ab5e8637e870250f/spec/json/twilio_api_v2010.json",
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in SPECS.items():
        r = httpx.get(url, timeout=120, follow_redirects=True)
        r.raise_for_status()
        (OUT / name).write_bytes(r.content)
        print(f"{name}: {len(r.content) / 1e6:.1f} MB")
