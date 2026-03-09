#!/usr/bin/env python3
"""
Run the evaluation test queries against the agent API and save full response
to execution_logs/. Each log file is structured for readability:

  1. query         — the question asked
  2. final_answer  — the agent's answer
  3. logs          — full trace (events, tool_calls, etc.) below

Requires the API server to be running and the Cyber Ireland 2022 PDF to be uploaded
with the given namespace (default: cyber-ireland-2022).

Usage:
  python scripts/run_test_queries_and_save_traces.py [--base-url URL] [--namespace NS]

Output:
  execution_logs/test1_verification.json
  execution_logs/test2_data_synthesis.json
  execution_logs/test3_forecasting.json
  execution_logs/test4_benchmarking.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Defaults
DEFAULT_BASE_URL = "http://localhost:8001"
DEFAULT_NAMESPACE = "cyber-ireland-2022"
EXECUTION_LOGS_DIR = Path(__file__).resolve().parent.parent / "execution_logs"

TEST_QUERIES = [
    {
        "id": "test1_verification",
        "query": "What is the total number of jobs reported, and where exactly is this stated?",
        "description": "Verification challenge: exact number + page and citation",
    },
    {
        "id": "test2_data_synthesis",
        "query": "Compare the concentration of 'Pure-Play' cybersecurity firms in the South-West against the National Average.",
        "description": "Data synthesis: regional tables, comparative metrics",
    },
    {
        "id": "test3_forecasting",
        "query": "Based on our 2022 baseline and the stated 2030 job target, what is the required compound annual growth rate (CAGR) to hit that goal?",
        "description": "Forecasting: baseline + target, CAGR via tool",
    },
    {
        "id": "test4_benchmarking",
        "query": "In the report's benchmarking section, how do Ireland, Northern Ireland, and the UK compare in terms of number of cyber security employees and sector GVA? Give the exact figures for each region and cite the table and page.",
        "description": "Complex: multi-region comparison, two metrics, exact numbers + citation",
    },
]


def run_query(base_url: str, namespace: str, query: str, max_steps: int = 10) -> dict:
    """POST to /agent/query with debug=true. Returns parsed JSON response."""
    url = f"{base_url.rstrip('/')}/agent/query"
    payload = {
        "query": query,
        "namespace": namespace,
        "max_steps": max_steps,
        "debug": True,
    }
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation test queries and save agent traces to execution_logs/."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("AGENT_BASE_URL", DEFAULT_BASE_URL),
        help=f"API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--namespace",
        default=os.environ.get("AGENT_NAMESPACE", DEFAULT_NAMESPACE),
        help=f"Namespace used when uploading the PDF (default: {DEFAULT_NAMESPACE})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Max agent steps per query (default: 10)",
    )
    args = parser.parse_args()

    EXECUTION_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    for spec in TEST_QUERIES:
        out_file = EXECUTION_LOGS_DIR / f"{spec['id']}.json"
        print(f"Running: {spec['id']} — {spec['query'][:60]}...")
        try:
            response = run_query(
                args.base_url,
                args.namespace,
                spec["query"],
                max_steps=args.max_steps,
            )
            # Readable order: question, final answer, then logs below
            resp_data = response if isinstance(response, dict) else {}
            export = {
                "test_id": spec["id"],
                "description": spec["description"],
                "query": spec["query"],
                "final_answer": resp_data.get("answer", ""),
                "status": resp_data.get("status", ""),
                "steps": resp_data.get("steps", 0),
                "run_at": datetime.utcnow().isoformat() + "Z",
                "namespace": args.namespace,
                "logs": resp_data,
            }
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(export, f, indent=2, ensure_ascii=False)
            print(f"  -> {out_file}")
        except HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            print(f"  ERROR HTTP {e.code}: {body[:200]}", file=sys.stderr)
            sys.exit(1)
        except URLError as e:
            print(f"  ERROR: {e.reason}", file=sys.stderr)
            print("  Is the server running? Try: python app.py", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\nDone. Traces saved under {EXECUTION_LOGS_DIR}. Commit these files for submission.")


if __name__ == "__main__":
    main()
