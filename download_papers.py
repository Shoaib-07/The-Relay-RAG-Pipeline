from pathlib import Path
from urllib.request import Request, urlopen

PAPERS = {
    "attention_is_all_you_need_v7.pdf": "https://arxiv.org/pdf/1706.03762v7",
    "moshi_v1.pdf": "https://arxiv.org/pdf/2410.00037v1",
    "dynamic_chunking_v2.pdf": "https://arxiv.org/pdf/2507.07955v2",
}

OUT = Path("data/papers")
OUT.mkdir(parents=True, exist_ok=True)

for filename, url in PAPERS.items():
    destination = OUT / filename
    if destination.exists() and destination.stat().st_size > 10_000:
        print(f"Already downloaded: {destination}")
        continue
    print(f"Downloading {filename} ...")
    request = Request(url, headers={"User-Agent": "TheRelayAssignment/1.0"})
    with urlopen(request, timeout=60) as response:
        content = response.read()
    if not content.startswith(b"%PDF"):
        raise RuntimeError(f"Downloaded file is not a PDF: {filename}")
    destination.write_bytes(content)
    print(f"Saved {destination} ({len(content):,} bytes)")

print("Paper download step finished.")
