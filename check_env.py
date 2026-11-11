import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pymupdf

load_dotenv()
print("Python:", sys.version.split()[0])
print("Working directory:", Path.cwd())
print("GROQ_API_KEY configured:", bool(os.getenv("GROQ_API_KEY")))
print("Groq model:", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

paper_dir = Path("data/papers")
pdfs = sorted(paper_dir.glob("*.pdf"))
print(f"PDF files found: {len(pdfs)}")
for pdf in pdfs:
    try:
        doc = pymupdf.open(pdf)
        first_page_text = doc[0].get_text()[:100].replace("\n", " ") if len(doc) else ""
        print(f"  OK: {pdf.name} | pages={len(doc)} | first-page text={first_page_text!r}")
    except Exception as exc:
        print(f"  ERROR: {pdf.name}: {exc}")

if len(pdfs) != 3:
    print("NEXT: run `python download_papers.py` to fetch the three required versions.")
