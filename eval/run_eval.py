import json
import os
import sys
from pathlib import Path

import httpx

SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:8000")
MIN_PASS_RATE = float(os.getenv("MIN_PASS_RATE", "0.8"))
CASES = Path(__file__).parent / "cases.jsonl"
RESULTS = Path(__file__).parent / "results.json"


def normalize(text: str) -> str:
    return text.lower().replace("\u2019", "'")


def main() -> int:
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    results = []
    with httpx.Client(timeout=180) as client:
        for c in cases:
            r = client.post(
                f"{SERVICE_URL}/ask",
                json={"question": c["question"], "context": c["context"]},
            )
            r.raise_for_status()
            data = r.json()
            answer = normalize(data["answer"])
            passed = any(k in answer for k in c["expect_any"])
            results.append({**c, **data, "passed": passed})
            print(f"{'PASS' if passed else 'FAIL'}  {c['id']}  {data['latency_ms']:>5} ms  {data['answer'][:70]}")

    n = len(results)
    pass_rate = sum(r["passed"] for r in results) / n
    summary = {
        "pass_rate": round(pass_rate, 3),
        "avg_prompt_tokens": round(sum(r["prompt_tokens"] for r in results) / n, 1),
        "avg_completion_tokens": round(sum(r["completion_tokens"] for r in results) / n, 1),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n),
        "prompt_version": results[0]["prompt_version"],
        "model": results[0]["model"],
    }
    RESULTS.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print("\nSUMMARY", json.dumps(summary))
    if pass_rate < MIN_PASS_RATE:
        print(f"GATE FAILED: pass rate {pass_rate:.0%} < {MIN_PASS_RATE:.0%}")
        return 1
    print(f"GATE PASSED: pass rate {pass_rate:.0%} >= {MIN_PASS_RATE:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
