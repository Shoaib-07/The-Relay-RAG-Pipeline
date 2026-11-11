from src.retriever import PaperRetriever

QUESTIONS = [
    "What is the Transformer architecture and how does it use attention?",
    "What is multi-head attention?",
    "How does Moshi handle real-time spoken dialogue?",
    "What are the main components of the Moshi speech-text model?",
    "What is dynamic chunking and why is it used for hierarchical sequence modeling?",
]

def main():
    retriever = PaperRetriever(top_k=3)
    print(f"Loaded {len(retriever.records)} indexed chunks.")
    for i, question in enumerate(QUESTIONS, start=1):
        print("\n" + "=" * 88)
        print(f"TEST {i}: {question}")
        results = retriever.retrieve(question, top_k=3)
        if not results:
            print("NO MATCH FOUND")
            continue
        for rank, item in enumerate(results, start=1):
            print(f"\n[{rank}] score={item['score']:.4f}")
            print(f"Source: {item['paper']}, page {item['page']}")
            print(item["text"][:650].replace("\n", " "))
    print("\nRetrieval-only test complete. No Groq API calls were made.")

if __name__ == "__main__":
    main()
