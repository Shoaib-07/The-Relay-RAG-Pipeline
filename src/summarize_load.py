
import csv
import math
from collections import defaultdict
from pathlib import Path

path = Path("logs/results.csv")
groups = defaultdict(list)

with path.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        groups[row["scenario"]].append(row)

def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    index = max(0, math.ceil(p * len(values)) - 1)
    return round(values[index], 2)

print("\nSCENARIO-WISE RESULTS")
print("-" * 110)

for scenario, rows in groups.items():
    latencies = [
        float(r["total_ms"]) for r in rows
        if r["total_ms"].strip()
    ]

    ttft = [
        float(r["first_token_ms"]) for r in rows
        if r["first_token_ms"].strip()
    ]

    counts = defaultdict(int)
    for r in rows:
        counts[r["status"]] += 1

    retries = [
        int(r["retries"]) for r in rows
        if r["retries"].strip()
    ]

    print(f"\nScenario: {scenario}")
    print("Requests:", len(rows))
    print("Success:", counts["ok"])
    print("Rate limited:", counts["rate_limited"])
    print("Timeout:", counts["timeout"])
    print("Errors:", counts["error"])
    print("Success %:", round(100 * counts["ok"] / len(rows), 2))
    print("Latency p50/p95/p99:",
          percentile(latencies, .50),
          percentile(latencies, .95),
          percentile(latencies, .99))
    print("TTFT p50/p95:",
          percentile(ttft, .50),
          percentile(ttft, .95))
    print("Average retries/request:",
          round(sum(retries) / len(retries), 3) if retries else 0)

print("\nPercentiles use the nearest-rank method.")
