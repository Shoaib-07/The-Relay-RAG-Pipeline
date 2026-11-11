import csv
import json
from pathlib import Path

index_path = Path("data/index/chunks_350.json")
results_path = Path("logs/retrieval/retrieval_350_rerank.csv")

chunks = json.loads(index_path.read_text(encoding="utf-8"))

with results_path.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

top_rows = [
    row for row in rows
    if row["reranked_rank"] == "1"
]

for row in top_rows:
    filename = row["retrieved_filename"]
    page = int(row["page"])

    matches = [
        c for c in chunks
        if c["filename"] == filename
        and int(c["page"]) == page
    ]

    print("\n" + "=" * 80)
    print(row["question_id"], "| Page:", page)
    print("Expected:", row["expected_filename"])
    print("Retrieved:", filename)

    if matches:
        for match in matches:
            print("\n--- Retrieved passage ---\n")
            print(match["text"])
    else:
        print("No matching passage found.")