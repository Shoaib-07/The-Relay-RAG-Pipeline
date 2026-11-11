
from pathlib import Path
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = Path("data/index")


class PaperRetriever:
    def __init__(
        self,
        chunk_words=350,
        top_k=5,
        rerank=False,
        candidate_k=10,
    ):
        index_path = INDEX_DIR / f"chunks_{chunk_words}.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"Index missing: {index_path}. "
                "Run the ingestion script for this chunk size."
            )

        self.records = json.loads(
            index_path.read_text(encoding="utf-8")
        )

        if not self.records:
            raise RuntimeError("The paper index is empty.")

        self.top_k = top_k
        self.rerank = rerank
        self.candidate_k = candidate_k

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=60000,
            sublinear_tf=True,
        )

        self.matrix = self.vectorizer.fit_transform(
            [record["text"] for record in self.records]
        )

    @staticmethod
    def _terms(text):
        return set(
            re.findall(
                r"\b[a-zA-Z][a-zA-Z0-9-]*\b",
                text.lower()
            )
        )

    def retrieve(self, question: str, top_k=None):
        k = top_k or self.top_k

        query_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(
            query_vec, self.matrix
        ).ravel()

        candidate_count = min(
            self.candidate_k if self.rerank else k,
            len(self.records)
        )

        candidate_indices = scores.argsort()[::-1][
            :candidate_count
        ]

        results = []

        query_terms = self._terms(question) - {
            "what", "how", "why", "does", "the", "and",
            "is", "are", "in", "for", "of", "to", "a",
            "an", "with", "from", "it", "this", "that",
        }

        max_score = max(
            (float(scores[i]) for i in candidate_indices),
            default=0.0,
        )

        for idx in candidate_indices:
            score = float(scores[idx])

            if score <= 0:
                continue

            item = dict(self.records[int(idx)])
            item["score"] = score

            if self.rerank:
                passage_terms = self._terms(item["text"])

                coverage = (
                    len(query_terms & passage_terms)
                    / len(query_terms)
                    if query_terms else 0.0
                )

                normalized_score = (
                    score / max(max_score, 1e-9)
                )

                item["term_coverage"] = coverage
                item["rerank_score"] = (
                    0.70 * score + 0.30 * coverage
                )

            results.append(item)

        if self.rerank:
            results.sort(
                key=lambda item: item["rerank_score"],
                reverse=True,
            )

        return results[:k]


def retrieval_tool(
    question: str,
    retriever: PaperRetriever,
    top_k=None,
):
    """Tool called by the pipeline to search the local paper library."""
    return retriever.retrieve(question, top_k=top_k)
