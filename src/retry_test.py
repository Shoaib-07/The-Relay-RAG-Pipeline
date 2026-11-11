
import asyncio
import csv
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.retry_client import RetryClient

CSV_PATH = Path("logs/results.csv")

FIELDS = [
    "request_id", "scenario", "source", "start_time",
    "first_token_ms", "total_ms", "status", "retries",
    "tokens_in", "tokens_out", "retrieval_setting", "answer_grade",
]


async def main():
    client = RetryClient(
        max_retries=2,
        base_backoff=0.5,
        timeout=8.0,
    )

    results = []

    for i in range(20):
        request_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()

        response = await client.request(
            f"Retry experiment question {i + 1}"
        )

        total_ms = round(
            (time.perf_counter() - start) * 1000, 2
        )

        results.append({
            "request_id": request_id,
            "scenario": "retry_experiment",
            "source": "stand-in",
            "start_time": start_time,
            # Non-streaming response: TTFT is not measured.
            "first_token_ms": "",
            "total_ms": total_ms,
            "status": response["status"],
            "retries": response["retries"],
            "tokens_in": "",
            "tokens_out": "",
            "retrieval_setting": "standin-default",
            "answer_grade": "",
        })

        print(
            f"{i + 1:02d}/20 status={response['status']} "
            f"retries={response['retries']} "
            f"total={total_ms} ms "
            f"history={response['history']}"
        )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0

    with CSV_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(results)

    durations = sorted(r["total_ms"] for r in results)

    def percentile(values, p):
        index = max(0, min(len(values) - 1, int(p * len(values) + 0.999) - 1))
        return values[index]

    print("\nRETRY EXPERIMENT SUMMARY")
    print("Requests:", len(results))
    print("Success:", sum(r["status"] == "ok" for r in results))
    print("Rate limited:", sum(r["status"] == "rate_limited" for r in results))
    print("Errors:", sum(r["status"] == "error" for r in results))
    print("Timeouts:", sum(r["status"] == "timeout" for r in results))
    print("Total retries:", sum(r["retries"] for r in results))
    print("Average retries/request:", round(
        statistics.mean(r["retries"] for r in results), 2
    ))
    print("Total latency p50:", percentile(durations, 0.50), "ms")
    print("Total latency p95:", percentile(durations, 0.95), "ms")
    print("CSV appended:", CSV_PATH.resolve())


if __name__ == "__main__":
    asyncio.run(main())
