
import argparse
import asyncio
import csv
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

URL = "http://127.0.0.1:8001/chat"
CSV_PATH = Path("logs/results.csv")

FIELDS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
]


async def send_request(client, scenario, semaphore=None):
    request_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    first_token_ms = ""
    status = "error"

    async def perform():
        nonlocal first_token_ms, status

        async with client.stream(
            "POST",
            URL,
            json={
                "question": "What is a Transformer?",
                "stream": True,
            },
        ) as response:

            if response.status_code == 429:
                status = "rate_limited"
            elif response.status_code == 503:
                status = "error"
            elif response.status_code == 504:
                status = "timeout"
            elif response.status_code == 200:
                status = "ok"
            else:
                status = "error"

            async for chunk in response.aiter_text():
                if chunk and not first_token_ms:
                    first_token_ms = round(
                        (time.perf_counter() - start) * 1000, 2
                    )

    try:
        if semaphore:
            async with semaphore:
                await perform()
        else:
            await perform()

    except httpx.TimeoutException:
        status = "timeout"
    except httpx.HTTPError as exc:
        print(f"Request error: {exc}")
        status = "error"

    total_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "request_id": request_id,
        "scenario": scenario,
        "source": "stand-in",
        "start_time": start_time,
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
        "status": status,
        "retries": 0,
        "tokens_in": "",
        "tokens_out": "",
        "retrieval_setting": "standin-default",
        "answer_grade": "",
    }


async def run_scenario(name, count, interval):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    timeout = httpx.Timeout(10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = []

        for i in range(count):
            tasks.append(asyncio.create_task(
                send_request(client, name)
            ))
            if interval > 0 and i < count - 1:
                await asyncio.sleep(interval)

        results = await asyncio.gather(*tasks)

    file_exists = CSV_PATH.exists()

    with CSV_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not file_exists or CSV_PATH.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(results)

    successful = sum(r["status"] == "ok" for r in results)
    limited = sum(r["status"] == "rate_limited" for r in results)
    timeouts = sum(r["status"] == "timeout" for r in results)
    errors = sum(r["status"] == "error" for r in results)

    durations = sorted(r["total_ms"] for r in results)

    def percentile(values, p):
        if not values:
            return None
        index = min(len(values) - 1, round((len(values) - 1) * p))
        return values[index]

    print(f"\nScenario: {name}")
    print(f"Requests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Rate limited: {limited}")
    print(f"Timeouts: {timeouts}")
    print(f"Errors: {errors}")
    print(f"Total time p50: {percentile(durations, 0.50)} ms")
    print(f"Total time p95: {percentile(durations, 0.95)} ms")
    print(f"CSV saved: {CSV_PATH.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scenario",
        choices=["steady", "burst", "overload", "half_rpm", "daily_budget"]
    )
    args = parser.parse_args()

    if args.scenario == "steady":
        asyncio.run(run_scenario("steady", 10, 1.0))
    elif args.scenario == "burst":
        asyncio.run(run_scenario("burst", 20, 0))
    elif args.scenario == "overload":
        asyncio.run(run_scenario("sustained_overload", 40, 0.1))
    elif args.scenario == "half_rpm":
        asyncio.run(run_scenario("half_rpm", 35, 0))
    else:
        asyncio.run(run_scenario("daily_budget_exhausted", 11, 0.2))
