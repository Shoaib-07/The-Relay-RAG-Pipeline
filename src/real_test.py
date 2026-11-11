
import os
import time
import csv
import uuid
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from src.groq_client import GroqModel

load_dotenv()

model = GroqModel()
output_file = Path("logs/real_groq_results.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)

questions = [
    "What is scaled dot-product attention?",
    "What is multi-head attention?",
    "Why does the Transformer use positional encoding?",
    "What is the role of Mimi in Moshi?",
    "What is dynamic chunking in hierarchical sequence modeling?",
]

columns = [
    "request_id", "scenario", "source", "start_time",
    "first_token_ms", "total_ms", "status", "retries",
    "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"
]

with output_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    for i, question in enumerate(questions, 1):
        request_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()

        try:
            result = model.answer(
                [
                    {
                        "role": "system",
                        "content": "Answer this question concisely in one or two sentences."
                    },
                    {"role": "user", "content": question},
                ],
                max_tokens=150,
                temperature=0.0,
            )

            total_ms = (time.perf_counter() - start) * 1000

            status = "ok" if result.get("text", "").strip() else "error"

            row = {
                "request_id": request_id,
                "scenario": "real_sequential",
                "source": "real",
                "start_time": start_time,
                "first_token_ms": "",
                "total_ms": round(total_ms, 2),
                "status": status,
                "retries": 0,
                "tokens_in": result.get("tokens_in", ""),
                "tokens_out": result.get("tokens_out", ""),
                "retrieval_setting": "direct_groq_no_retrieval",
                "answer_grade": "",
            }

            print(f"\nRequest {i}: {status}")
            print("Latency:", round(total_ms, 2), "ms")
            print("Tokens:", row["tokens_in"], "in /", row["tokens_out"], "out")
            print("Answer:", result.get("text", "")[:250])

        except Exception as exc:
            total_ms = (time.perf_counter() - start) * 1000
            row = {
                "request_id": request_id,
                "scenario": "real_sequential",
                "source": "real",
                "start_time": start_time,
                "first_token_ms": "",
                "total_ms": round(total_ms, 2),
                "status": "error",
                "retries": 0,
                "tokens_in": "",
                "tokens_out": "",
                "retrieval_setting": "direct_groq_no_retrieval",
                "answer_grade": "",
            }
            print(f"\nRequest {i}: ERROR - {repr(exc)}")

        writer.writerow(row)
        f.flush()

        if i < len(questions):
            time.sleep(2)

print("\nSaved results to:", output_file)
