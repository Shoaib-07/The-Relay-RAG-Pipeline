"""
Run a 15-question answer-quality evaluation for three retrieval settings.
This makes 45 real Groq pipeline calls. It spaces model calls to stay comfortably
below the dashboard's 8,000 TPM limit. TTFT is not measured because the current
Groq client is non-streaming.
"""
import csv
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.pipeline import RelayPipeline

QUESTIONS = [
    ("A01", "What is the Transformer architecture and how does it differ from recurrent models?", "attention_is_all_you_need_v7.pdf"),
    ("A02", "How does scaled dot-product attention calculate attention weights?", "attention_is_all_you_need_v7.pdf"),
    ("A03", "Why does the Transformer use multi-head attention?", "attention_is_all_you_need_v7.pdf"),
    ("A04", "What are positional encodings and why are they needed?", "attention_is_all_you_need_v7.pdf"),
    ("A05", "How does the Transformer encoder and decoder work?", "attention_is_all_you_need_v7.pdf"),
    ("M01", "What is Moshi and what problem does it address in real-time dialogue?", "moshi_v1.pdf"),
    ("M02", "How does Moshi handle speech and text in real-time conversation?", "moshi_v1.pdf"),
    ("M03", "What is the role of the Mimi audio codec in Moshi?", "moshi_v1.pdf"),
    ("M04", "How does Moshi reduce latency during speech generation?", "moshi_v1.pdf"),
    ("M05", "What are the main components of the Moshi architecture?", "moshi_v1.pdf"),
    ("D01", "What is dynamic chunking in end-to-end hierarchical sequence modeling?", "dynamic_chunking_v2.pdf"),
    ("D02", "Why is hierarchical sequence modeling useful for long sequences?", "dynamic_chunking_v2.pdf"),
    ("D03", "How are chunks selected or formed in dynamic chunking?", "dynamic_chunking_v2.pdf"),
    ("D04", "What problem does dynamic chunking solve compared with fixed chunking?", "dynamic_chunking_v2.pdf"),
    ("D05", "What are the main results or benefits reported for dynamic chunking?", "dynamic_chunking_v2.pdf"),
]

# Three distinct retrieval settings: vary both chunk size and number fetched.
SETTINGS = [
    {"name": "150w_top3_no_rerank", "chunk_words": 150, "top_k": 3, "rerank": False},
    {"name": "220w_top5_no_rerank", "chunk_words": 220, "top_k": 5, "rerank": False},
    {"name": "350w_top5_no_rerank", "chunk_words": 350, "top_k": 5, "rerank": False},
]

OUT = Path("logs/retrieval/answer_quality_results.csv")
RAW_RESULTS = Path("logs/results.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

DETAIL_FIELDS = [
    "request_id", "setting", "chunk_words", "top_k", "rerank",
    "question_id", "question", "expected_paper", "status",
    "total_ms", "first_token_ms", "tokens_in", "tokens_out",
    "answer_grade", "answer", "evidence"
]
RESULT_FIELDS = [
    "request_id", "scenario", "source", "start_time",
    "first_token_ms", "total_ms", "status", "retries",
    "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"
]

def main():
    total_calls = len(SETTINGS) * len(QUESTIONS)
    print(f"Starting {total_calls} real Groq pipeline evaluations.")
    print("Estimated duration: about 25 minutes at a 35-second spacing.")
    print("Do not share or print your .env/API key.")
    print("TTFT will be blank: current Groq calls are non-streaming.\n")

    with OUT.open("w", newline="", encoding="utf-8") as detail_file:
        detail_writer = csv.DictWriter(detail_file, fieldnames=DETAIL_FIELDS)
        detail_writer.writeheader()

        with RAW_RESULTS.open("a", newline="", encoding="utf-8") as results_file:
            results_writer = csv.DictWriter(results_file, fieldnames=RESULT_FIELDS)

            completed = 0
            for setting in SETTINGS:
                print(f"\n=== Setting: {setting['name']} ===")
                pipeline = RelayPipeline(
                    top_k=setting["top_k"],
                    chunk_words=setting["chunk_words"],
                    rerank=setting["rerank"],
                )

                for qid, question, expected in QUESTIONS:
                    request_id = str(uuid.uuid4())
                    start_time = datetime.now(timezone.utc).isoformat()
                    start = time.perf_counter()
                    output = {}
                    status = "error"
                    tokens_in = ""
                    tokens_out = ""
                    answer = ""
                    evidence_text = ""

                    try:
                        output = pipeline.run(question)
                        status = "ok" if output.get("status") == "ok" else "refused"
                        tokens_in = output.get("tokens_in", 0)
                        tokens_out = output.get("tokens_out", 0)
                        answer = output.get("answer", "")
                        evidence_text = " || ".join(
                            f"{e.get('paper')} p.{e.get('page')} score={e.get('score')}"
                            for e in output.get("evidence", [])
                        )
                    except Exception as exc:
                        answer = f"EXCEPTION: {type(exc).__name__}: {exc}"
                        evidence_text = ""

                    total_ms = round((time.perf_counter() - start) * 1000, 2)
                    completed += 1

                    detail_writer.writerow({
                        "request_id": request_id,
                        "setting": setting["name"],
                        "chunk_words": setting["chunk_words"],
                        "top_k": setting["top_k"],
                        "rerank": setting["rerank"],
                        "question_id": qid,
                        "question": question,
                        "expected_paper": expected,
                        "status": status,
                        "total_ms": total_ms,
                        "first_token_ms": "",
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "answer_grade": "",
                        "answer": answer,
                        "evidence": evidence_text,
                    })
                    detail_file.flush()

                    results_writer.writerow({
                        "request_id": request_id,
                        "scenario": f"answer_quality_{setting['name']}",
                        "source": "real",
                        "start_time": start_time,
                        "first_token_ms": "",
                        "total_ms": total_ms,
                        "status": "ok" if status in ("ok", "refused") else "error",
                        "retries": 0,
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "retrieval_setting": setting["name"],
                        "answer_grade": "",
                    })
                    results_file.flush()

                    print(
                        f"[{completed}/{total_calls}] {setting['name']} {qid}: "
                        f"{status}, {total_ms} ms, tokens {tokens_in}/{tokens_out}"
                    )

                    # 35 seconds spaces requests below the observed 8k TPM quota.
                    if completed < total_calls:
                        time.sleep(35)

    print("\nEvaluation calls finished.")
    print(f"Detailed answers: {OUT}")
    print(f"Raw request rows appended to: {RAW_RESULTS}")
    print("Next: manually grade each answer as correct, wrong, refused_correctly, or refused_wrongly.")

if __name__ == "__main__":
    main()
