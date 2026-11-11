
import json
from pathlib import Path

from src.ingestion import build_index

INDEX_DIR = Path("data/index")
CONFIGURATIONS = [
    {"chunk_words": 150, "overlap_words": 30},
    {"chunk_words": 220, "overlap_words": 40},
    {"chunk_words": 350, "overlap_words": 60},
]

for config in CONFIGURATIONS:
    size = config["chunk_words"]

    print(f"\nBuilding {size}-word index...")

    records = build_index(
        chunk_words=size,
        overlap_words=config["overlap_words"],
    )

    output = INDEX_DIR / f"chunks_{size}.json"
    output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {output}")
    print(f"Total chunks: {len(records)}")

print("\nAll retrieval indexes created.")
