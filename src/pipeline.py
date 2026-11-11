import re
from src.retriever import PaperRetriever, retrieval_tool
from src.groq_client import GroqModel

SYSTEM_PROMPT = """You answer engineering questions using ONLY the supplied evidence excerpts.
Rules:
- Do not use outside knowledge or memory.
- If the evidence is insufficient, say: "The provided papers do not establish this."
- Every factual claim must end with one or more evidence tags such as [E1] or [E2].
- Use ONLY evidence tags that appear in the supplied excerpts.
- Do not write paper titles, page numbers, or any other citation format yourself.
- If no excerpt supports the answer, refuse rather than guess.
- Be concise and directly answer the question.
"""

class RelayPipeline:
    def __init__(self, top_k=5, chunk_words=350, rerank=False):
        self.retriever = PaperRetriever(
            chunk_words=chunk_words,
            top_k=top_k,
            rerank=rerank
        )
        self.model = GroqModel()

    def run(self, question: str):
        question = question.strip()
        if len(question) < 5:
            return {
                "status": "refused",
                "answer": "Please provide a more specific question.",
                "tokens_in": 0, "tokens_out": 0, "evidence": []
            }

        # Step 1: retrieval tool call
        evidence = retrieval_tool(question, self.retriever)
        if not evidence:
            return {
                "status": "refused",
                "answer": "The provided papers do not establish this.",
                "tokens_in": 0, "tokens_out": 0, "evidence": []
            }

        # Step 2: conservative retrieval branch
        if evidence[0]["score"] < 0.025:
            return {
                "status": "refused",
                "answer": "The provided papers do not establish this.",
                "tokens_in": 0, "tokens_out": 0, "evidence": evidence
            }

        # Step 3: label evidence deterministically; model cannot invent page citations
        evidence_map = {}
        evidence_blocks = []
        for i, item in enumerate(evidence, start=1):
            tag = f"E{i}"
            evidence_map[tag] = item
            evidence_blocks.append(
                f"[{tag}] Source: {item['paper']}, page {item['page']}\n"
                f"Passage: {item['text']}"
            )
        evidence_text = "\n\n".join(evidence_blocks)
        user_prompt = (
            f"Question: {question}\n\n"
            f"Evidence excerpts:\n{evidence_text}\n\n"
            "Answer using only these excerpts. Put an evidence tag such as [E1] "
            "at the end of every factual claim. Do not invent tags."
        )

        # Step 4: real hosted model call
        result = self.model.answer([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ], max_tokens=500, temperature=0.0)

        raw_answer = result["text"]
        print("\nDEBUG - RAW GROQ RESPONSE:")
        print(repr(raw_answer))
        cited_tags = re.findall(r"\[(E\d+)\]", raw_answer)
        allowed_tags = set(evidence_map)
        invalid_tags = [tag for tag in cited_tags if tag not in allowed_tags]

        # Step 5: verify tags and convert them to exact paper/page citations
        if not cited_tags or invalid_tags:
            refusal_phrases = [
                "the provided papers do not establish this",
                "the evidence is insufficient",
                "i cannot determine this from the provided papers",
            ]

            if any(phrase in raw_answer.lower() for phrase in refusal_phrases):
                answer = "The provided papers do not establish this."
            else:
                answer = (
                    "I cannot provide a verified cited answer from the retrieved passages. "
                    "The answer was withheld because its evidence tags were missing or invalid."
                )

            status = "refused"
        else:
            answer = re.sub(
                r"\[(E\d+)\]",
                lambda match: (
                    f"[{evidence_map[match.group(1)]['paper']}, "
                    f"p. {evidence_map[match.group(1)]['page']}]"
                    if match.group(1) in evidence_map else match.group(0)
                ),
                raw_answer,
            )
            status = "ok"

        return {
            "status": status,
            "answer": answer,
            "tokens_in": result["tokens_in"],
            "tokens_out": result["tokens_out"],
            "evidence": evidence,
            "model": result["model"],
            "best_retrieval_score": evidence[0]["score"],
            "raw_model_answer": raw_answer,
        }

if __name__ == "__main__":
    pipeline = RelayPipeline()
    while True:
        question = input("\nQuestion (or 'exit'): ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        try:
            output = pipeline.run(question)
            print("\nSTATUS:", output["status"])
            print("\nANSWER:\n", output["answer"])
            print("\nTOKENS:", output["tokens_in"], "in /", output["tokens_out"], "out")
            print("\nEVIDENCE:")
            for item in output.get("evidence", []):
                print(f"- {item['paper']}, p. {item['page']} (score={item['score']:.3f})")
        except Exception as exc:
            print(f"Pipeline error: {type(exc).__name__}: {exc}")
