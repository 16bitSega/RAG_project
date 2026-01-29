import json
import os
import re
import hashlib
from pathlib import Path
from datetime import datetime

RAW_PDF_DIR = Path("data/raw_pdfs")
MCP_DIR = Path("mcp")
OUT_DIR = Path(os.environ.get("RAG_OUT_DIR", "data/normalized"))
SOURCES = Path("sources.json")

# -------- PDF extraction --------
def extract_text_pypdf(pdf_path: Path) -> list[str]:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return pages

def extract_text_pdfminer(pdf_path: Path) -> list[str]:
    from pdfminer.high_level import extract_text
    text = extract_text(str(pdf_path)) or ""
    return [text]

def extract_pages(pdf_path: Path) -> list[str]:
    try:
        pages = extract_text_pypdf(pdf_path)
        nonempty = sum(1 for p in pages if p.strip())
        if nonempty < max(1, len(pages) // 10):
            return extract_text_pdfminer(pdf_path)
        return pages
    except Exception:
        return extract_text_pdfminer(pdf_path)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# -------- normalization + chunking --------
HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
MULTI_NL = re.compile(r"\n{3,}")
WS = re.compile(r"[ \t]+")

def normalize_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = HYPHEN_BREAK.sub(r"\1\2", s)
    s = WS.sub(" ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = MULTI_NL.sub("\n\n", s)
    return s.strip()

def chunk_text(text: str, target_chars: int = 2400, overlap_chars: int = 300) -> list[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= target_chars:
            buf += "\n\n" + p
        else:
            chunks.append(buf)
            tail = buf[-overlap_chars:] if overlap_chars and len(buf) > overlap_chars else ""
            buf = (tail + "\n\n" + p).strip() if tail else p
    if buf:
        chunks.append(buf)

    # window oversized chunks
    out = []
    for c in chunks:
        if len(c) <= target_chars * 2:
            out.append(c)
        else:
            step = max(1, target_chars - overlap_chars)
            for i in range(0, len(c), step):
                part = c[i:i + target_chars].strip()
                if part:
                    out.append(part)
    return out

# Best-effort heading split for PDFs
SECTION_HEADING = re.compile(r"^(?:[A-Z][A-Z0-9 /,-]{6,}|(?:\d+(?:\.\d+){0,3})\s+[A-Z]).*$")
CHAPTER_HEADING = re.compile(r"^(?:CHAPTER\s+\d+|Chapter\s+\d+|\d+\s+CHAPTER)\b")

STOPWORDS = {
    "a","an","and","are","as","at","be","but","by","can","do","does","for","from","how","i","if","in","is","it","of","on","or",
    "that","the","their","then","there","these","this","to","was","were","what","when","where","which","who","why","with","you","your"
}

def sentence_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def summarize_text(text: str, max_sentences: int = 3, max_chars: int = 800) -> str:
    sentences = sentence_split(text)
    summary = " ".join(sentences[:max_sentences]).strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0].strip()
    return summary

def extract_tags(text: str, title: str | None, section_title: str | None, max_tags: int = 8) -> list[str]:
    content = " ".join([t for t in [title, section_title, text] if t])
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", content)
    lowered = [t.lower() for t in tokens if t.lower() not in STOPWORDS]
    freq = {}
    for t in lowered:
        freq[t] = freq.get(t, 0) + 1
    keywords = sorted(freq.keys(), key=lambda k: (-freq[k], k))[:max_tags]

    entities = []
    for m in re.findall(r"\b[A-Z][a-zA-Z]+\b(?:\s+[A-Z][a-zA-Z]+\b){0,2}", content):
        ent = m.strip()
        if ent.lower() in STOPWORDS:
            continue
        if ent not in entities:
            entities.append(ent)
        if len(entities) >= max_tags:
            break

    tags = []
    for k in keywords + entities:
        if k and k not in tags:
            tags.append(k)
    return tags[:max_tags]

def build_breadcrumbs(doc_title: str, section_title: str | None) -> str:
    if section_title:
        return f"Book: {doc_title} > Section: {section_title}"
    return f"Book: {doc_title}"

def split_by_headings(pages: list[str]) -> list[dict]:
    blocks = []
    current_title = None
    current = []
    start_page = 1

    for idx, page in enumerate(pages, start=1):
        lines = [ln.rstrip() for ln in page.split("\n")]
        for ln in lines:
            if SECTION_HEADING.match(ln.strip()) and len(ln.strip()) < 140:
                if current:
                    blocks.append({
                        "title": current_title,
                        "text": normalize_text("\n".join(current)),
                        "page_start": start_page,
                        "page_end": idx
                    })
                    current = []
                current_title = ln.strip()
                start_page = idx
            else:
                current.append(ln)
    if current:
        blocks.append({
            "title": current_title,
            "text": normalize_text("\n".join(current)),
            "page_start": start_page,
            "page_end": len(pages)
        })

    pruned = [b for b in blocks if len(b["text"]) >= 400]
    return pruned

# MCP markdown split: chunk by headings to keep semantics
MD_H1 = re.compile(r"(?m)^#\s+")

def split_markdown(md: str) -> list[dict]:
    md = md.strip()
    if not md:
        return []
    # Split on H1 headings but keep first if no heading
    if "\n# " not in "\n" + md:
        return [{"title": None, "text": normalize_text(md)}]

    blocks = []
    current_title = None
    current = []
    for line in md.splitlines():
        if line.startswith("# "):
            if current:
                blocks.append({"title": current_title, "text": normalize_text("\n".join(current))})
                current = []
            current_title = line[2:].strip() or None
        else:
            current.append(line)
    if current:
        blocks.append({"title": current_title, "text": normalize_text("\n".join(current))})
    return [b for b in blocks if len(b["text"]) >= 200]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))["sources"]

    out_jsonl = OUT_DIR / "chunks_books.jsonl"
    out_jsonl.write_text("", encoding="utf-8")

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "documents": []
    }

    chunk_counter = 0

    # Ingest PDFs defined in sources.json
    for s in sources:
        if s.get("format") != "pdf":
            continue
        pdf_path = RAW_PDF_DIR / s["filename"]
        if not pdf_path.exists():
            print(f"[WARN] Missing PDF: {pdf_path}")
            continue

        pages = extract_pages(pdf_path)
        blocks = split_by_headings(pages)
        if not blocks:
            blocks = []
            for i, p in enumerate(pages, start=1):
                t = normalize_text(p)
                if len(t) >= 400:
                    blocks.append({"title": None, "text": t, "page_start": i, "page_end": i})

        manifest["documents"].append({
            "id": s["id"],
            "title": s["title"],
            "format": "pdf",
            "filename": s["filename"],
            "sha256": sha256_file(pdf_path),
            "blocks": len(blocks),
            "source_type": "book",
            "author": s.get("author"),
            "date": s.get("date")
        })

        for b in blocks:
            chunks = chunk_text(b["text"], target_chars=2400, overlap_chars=300)
            section_title = b.get("title")
            breadcrumbs = build_breadcrumbs(s["title"], section_title)
            summary = summarize_text(b["text"])
            summary_level = "chapter" if section_title and CHAPTER_HEADING.search(section_title) else "section"
            summary_tags = extract_tags(summary, s["title"], section_title)
            summary_rec = {
                "chunk_id": f"{s['id']}::summary::{chunk_counter + 1:06d}",
                "doc_id": s["id"],
                "doc_title": s["title"],
                "title": s["title"],
                "author": s.get("author"),
                "date": s.get("date"),
                "source_type": "book",
                "format": "pdf",
                "section_title": section_title,
                "page_start": b.get("page_start"),
                "page_end": b.get("page_end"),
                "breadcrumbs": breadcrumbs,
                "chunk_type": "summary",
                "summary_level": summary_level,
                "priority": 3,
                "tags": summary_tags,
                "url": None,
                "text": f"Breadcrumbs: {breadcrumbs}\nSummary ({summary_level}): {summary}"
            }
            if summary:
                chunk_counter += 1
                with out_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(summary_rec, ensure_ascii=False) + "\n")

            for c in chunks:
                chunk_counter += 1
                tags = extract_tags(c, s["title"], section_title)
                rec = {
                    "chunk_id": f"{s['id']}::{chunk_counter:06d}",
                    "doc_id": s["id"],
                    "doc_title": s["title"],
                    "title": s["title"],
                    "author": s.get("author"),
                    "date": s.get("date"),
                    "source_type": "book",
                    "format": "pdf",
                    "section_title": section_title,
                    "page_start": b.get("page_start"),
                    "page_end": b.get("page_end"),
                    "breadcrumbs": breadcrumbs,
                    "chunk_type": "section",
                    "priority": 2,
                    "tags": tags,
                    "url": None,
                    "text": f"Breadcrumbs: {breadcrumbs}\n{c}"
                }
                with out_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[OK] {s['id']}: {len(blocks)} blocks")

    # Ingest MCP markdown files
    if MCP_DIR.exists():
        for md_path in sorted(MCP_DIR.glob("*.md")):
            md_text = md_path.read_text(encoding="utf-8", errors="ignore")
            blocks = split_markdown(md_text)
            doc_id = f"mcp::{md_path.stem}"
            manifest["documents"].append({
                "id": doc_id,
                "title": f"MCP - {md_path.name}",
                "format": "markdown",
                "filename": str(md_path),
                "blocks": len(blocks),
                "source_type": "mcp",
                "author": None,
                "date": None
            })
            for b in blocks:
                chunks = chunk_text(b["text"], target_chars=1600, overlap_chars=120)
                section_title = b.get("title")
                breadcrumbs = f"MCP: {md_path.name}" + (f" > Section: {section_title}" if section_title else "")
                for c in chunks:
                    chunk_counter += 1
                    tags = extract_tags(c, f"MCP - {md_path.name}", section_title)
                    rec = {
                        "chunk_id": f"{doc_id}::{chunk_counter:06d}",
                        "doc_id": doc_id,
                        "doc_title": f"MCP - {md_path.name}",
                        "title": f"MCP - {md_path.name}",
                        "author": None,
                        "date": None,
                        "source_type": "mcp",
                        "format": "markdown",
                        "section_title": section_title,
                        "page_start": None,
                        "page_end": None,
                        "breadcrumbs": breadcrumbs,
                        "chunk_type": "section",
                        "priority": 2,
                        "tags": tags,
                        "url": None,
                        "text": f"Breadcrumbs: {breadcrumbs}\n{c}"
                    }
                    with out_jsonl.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[OK] MCP: ingested markdown from {MCP_DIR}")

    (OUT_DIR / "manifest_books.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone: {out_jsonl} and {OUT_DIR/'manifest_books.json'}")


if __name__ == "__main__":
    main()
