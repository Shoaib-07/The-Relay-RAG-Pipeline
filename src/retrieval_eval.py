
import csv
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = Path("data/index")
OUTPUT_DIR = Path("logs/retrieval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

STOPWORDS = {
    "what", "how", "why", "does", "the", "and", "is", "are",
    "in", "for", "of", "to", "a", "an", "with", "from",
    "it", "this", "that", "on", "by", "or", "as",
}


def terms(text):
    return set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]*\b", text.lower()))


def evaluate(index_path, chunk_size):
    records = json.loads(index_path.read_text(encoding="utf-8"))
    texts = [r["text"] for r in records]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=60000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)

    output_path = OUTPUT_DIR / f"retrieval_{chunk_size}_comparison.csv"

    summaries = {
        "tfidf": 0,
        "old_heuristic": 0,
        "normalized_heuristic": 0,
        "top5_recall": 0,
    }

    fields = [
        "question_id", "question", "expected_filename",
        "method", "rank", "retrieved_filename", "page",
        "tfidf_score", "term_coverage", "final_score",
        "top1_paper_match",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for qid, question, expected in QUESTIONS:
            query_vec = vectorizer.transform([question])
            scores = cosine_similarity(query_vec, matrix).ravel()

            candidates = scores.argsort()[::-1][:10]

            query_terms = terms(question) - STOPWORDS
            ranked = []

            for idx in candidates:
                passage_terms = terms(records[int(idx)]["text"])

                coverage = (
                    len(query_terms & passage_terms) / len(query_terms)
                    if query_terms else 0.0
                )

                ranked.append({
                    "idx": int(idx),
                    "tfidf": float(scores[idx]),
                    "coverage": coverage,
                })

            methods = {
                "tfidf": sorted(
                    ranked,
                    key=lambda x: x["tfidf"],
                    reverse=True,
                ),
                "old_heuristic": sorted(
                    ranked,
                    key=lambda x: 0.70 * x["tfidf"] + 0.30 * x["coverage"],
                    reverse=True,
                ),
                "normalized_heuristic": sorted(
                    ranked,
                    key=lambda x: 0.80 * (
                        x["tfidf"] / max(
                            max(item["tfidf"] for item in ranked), 1e-9
                        )
                    ) + 0.20 * x["coverage"],
                    reverse=True,
                ),
            }

            # Baseline top-5 expected-paper recall
            baseline_top5 = methods["tfidf"][:5]
            if any(
                records[item["idx"]]["filename"] == expected
                for item in baseline_top5
            ):
                summaries["top5_recall"] += 1

            for method, items in methods.items():
                top_record = records[items[0]["idx"]]
                match = top_record["filename"] == expected

                summaries[method] += int(match)

                for rank, item in enumerate(items[:5], start=1):
                    record = records[item["idx"]]

                    if method == "tfidf":
                        final_score = item["tfidf"]
                    elif method == "old_heuristic":
                        final_score = (
                            0.70 * item["tfidf"]
                            + 0.30 * item["coverage"]
                        )
                    else:
                        max_score = max(
                            x["tfidf"] for x in ranked
                        )
                        normalized_tfidf = (
                            item["tfidf"] / max(max_score, 1e-9)
                        )
                        final_score = (
                            0.80 * normalized_tfidf
                            + 0.20 * item["coverage"]
                        )

                    writer.writerow({
                        "question_id": qid,
                        "question": question,
                        "expected_filename": expected,
                        "method": method,
                        "rank": rank,
                        "retrieved_filename": record["filename"],
                        "page": record["page"],
                        "tfidf_score": round(item["tfidf"], 4),
                        "term_coverage": round(item["coverage"], 4),
                        "final_score": round(final_score, 4),
                        "top1_paper_match": match,
                    })

    total = len(QUESTIONS)

    print(f"\nChunk size: {chunk_size}")
    for method in (
        "tfidf", "old_heuristic", "normalized_heuristic"
    ):
        count = summaries[method]
        print(
            f"{method}: {count}/{total} "
            f"({count / total * 100:.1f}%)"
        )

    print(
        f"TF-IDF top-5 paper recall: "
        f"{summaries['top5_recall']}/{total}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    evaluate(INDEX_DIR / "chunks_150.json", 150)
    evaluate(INDEX_DIR / "chunks_220.json", 220)
    evaluate(INDEX_DIR / "chunks_350.json", 350)
