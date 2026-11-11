from pathlib import Path
import json
import re
import pymupdf

PAPER_DIR = Path("data/papers")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

PAPER_NAMES = {
    "attention_is_all_you_need_v7.pdf": "Attention Is All You Need (arXiv:1706.03762v7)",
    "moshi_v1.pdf": "Moshi: a speech-text foundation model for real-time dialogue (arXiv:2410.00037v1)",
    "dynamic_chunking_v2.pdf": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (arXiv:2507.07955v2)",
}

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, chunk_words: int = 220, overlap_words: int = 40):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) >= 40:
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap_words
    return chunks

def build_index(chunk_words: int = 220, overlap_words: int = 40):
    records = []
    pdfs = sorted(PAPER_DIR.glob("*.pdf"))
    if len(pdfs) != 3:
        raise RuntimeError(f"Expected 3 PDFs in {PAPER_DIR}, found {len(pdfs)}")

    for pdf_path in pdfs:
        paper_title = PAPER_NAMES.get(pdf_path.name, pdf_path.stem)
        doc = pymupdf.open(pdf_path)
        for page_index, page in enumerate(doc):
            text = clean_text(page.get_text("text"))
            if not text:
                continue
            for chunk_no, chunk in enumerate(chunk_text(text, chunk_words, overlap_words), start=1):
                records.append({
                    "chunk_id": f"{pdf_path.stem}-p{page_index + 1}-c{chunk_no}",
                    "paper": paper_title,
                    "filename": pdf_path.name,
                    "page": page_index + 1,
                    "chunk": chunk_no,
                    "text": chunk,
                })
        doc.close()

    output = INDEX_DIR / "chunks.json"
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Indexed {len(records)} chunks from {len(pdfs)} papers.")
    print(f"Saved: {output}")
    return records

if __name__ == "__main__":
    build_index()
