#!/usr/bin/env python3
import os
import re
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from readability import Document

# PDF fallback for arXiv / PDFs
from pdfminer.high_level import extract_text as pdfminer_extract_text


# -----------------------------
# Output
# -----------------------------
OUT_DIR = os.environ.get("RAG_OUT_DIR", "data/normalized")
OUT_JSONL = os.path.join(OUT_DIR, "chunks_articles.jsonl")
OUT_MANIFEST = os.path.join(OUT_DIR, "manifest_articles.json")


# -----------------------------
# Fetch config
# -----------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

TIMEOUT_S = 30


# -----------------------------
# Sources (latest recommendations)
# -----------------------------

# -----------------------------
# Sources file (recommended)
# -----------------------------
SOURCES_FILE = os.environ.get("RAG_ARTICLE_SOURCES", "sources_articles.json")

def load_sources() -> List[Dict]:
    # Prefer JSON config so users can add sources without editing code.
    p = Path(SOURCES_FILE)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{SOURCES_FILE} must be a JSON list of sources")
        return data
    return []

SOURCES: List[Dict] = load_sources() or [
    {
        "id": "anthropic_multi_agent_research_system",
        "type": "html",
        "publisher": "Anthropic",
        "url": "https://www.anthropic.com/engineering/multi-agent-research-system",
    },
    {
        "id": "anthropic_agentic_misalignment",
        "type": "html",
        "publisher": "Anthropic",
        "url": "https://www.anthropic.com/research/agentic-misalignment",
    },
    {
        "id": "react_arxiv_2210_03629",
        "type": "pdf",
        "publisher": "arXiv",
        "url": "https://arxiv.org/pdf/2210.03629.pdf",
    },
    {
        "id": "rag_arxiv_2005_11401",
        "type": "pdf",
        "publisher": "arXiv",
        "url": "https://arxiv.org/pdf/2005.11401.pdf",
    },
    {
        "id": "toolformer_arxiv_2302_04761",
        "type": "pdf",
        "publisher": "arXiv",
        "url": "https://arxiv.org/pdf/2302.04761.pdf",
    },
    {
        "id": "tds_single_vs_multi_agent_systems",
        "type": "html",
        "publisher": "Towards Data Science",
        "url": "https://towardsdatascience.com/agentic-ai-single-vs-multi-agent-systems/",
    },
    {
        "id": "tds_langgraph_101_deep_research_agent",
        "type": "html",
        "publisher": "Towards Data Science",
        "url": "https://towardsdatascience.com/langgraph-101-lets-build-a-deep-research-agent/",
    },
    {
        "id": "tds_effective_ai_agents_at_scale",
        "type": "html",
        "publisher": "Towards Data Science",
        "url": "https://towardsdatascience.com/how-to-build-effective-ai-agents-to-process-millions-of-requests/",
    },
    {
        "id": "ai_sdk_mcp_tools",
        "type": "html",
        "publisher": "AI SDK",
        "url": "https://ai-sdk.dev/docs/ai-sdk-core/mcp-tools"
    },
    {
        "id": "byteplus_mcp_topic",
        "type": "html",
        "publisher": "BytePlus",
        "url": "https://www.byteplus.com/en/topic/542256?title="
    },
    {
        "id": "merge_mcp_tool_schema",
        "type": "html",
        "publisher": "Merge.dev",
        "url": "https://www.merge.dev/blog/mcp-tool-schema"
    },
    {
        "id": "netfoundry_ai_agent_mcp_decision",
        "type": "html",
        "publisher": "NetFoundry",
        "url": "https://netfoundry.io/ai/how-an-ai-agent-decides-to-call-mcp-tools/"
    },
    {
        "id": "modelcontextprotocol_github",
        "type": "html",
        "publisher": "Model Context Protocol",
        "url": "https://github.com/modelcontextprotocol/modelcontextprotocol"
    },
    {
        "id": "devto_react_vs_plan_execute",
        "type": "html",
        "publisher": "Dev.to",
        "url": "https://dev.to/jamesli/react-vs-plan-and-execute-a-practical-comparison-of-llm-agent-patterns-4gh9"
    },
    {
        "id": "byaiteam_agent_planning_reliability",
        "type": "html",
        "publisher": "By AI Team",
        "url": "https://byaiteam.com/blog/2025/12/09/ai-agent-planning-react-vs-plan-and-execute-for-reliability/"
    },
    {
        "id": "linkedin_build_ai_agent_post",
        "type": "html",
        "publisher": "LinkedIn",
        "url": "https://www.linkedin.com/posts/lewisowain_how-to-build-an-ai-agent-activity-7402339630764941312-_G5h/"
    },
    {
        "id": "scitepress_multiagent_paper_2021",
        "type": "pdf",
        "publisher": "SciTePress",
        "url": "https://www.scitepress.org/Papers/2021/105593/105593.pdf"
    },
    {
        "id": "geeksforgeeks_informed_vs_uninformed_search",
        "type": "html",
        "publisher": "GeeksforGeeks",
        "url": "https://www.geeksforgeeks.org/artificial-intelligence/difference-between-informed-and-uninformed-search-in-ai/"
    },
    {
        "id": "baeldung_informed_vs_uninformed_search",
        "type": "html",
        "publisher": "Baeldung",
        "url": "https://www.baeldung.com/cs/informed-vs-uninformed-search"
    },
    {
        "id": "scaler_informed_vs_uninformed_search",
        "type": "html",
        "publisher": "Scaler",
        "url": "https://www.scaler.com/topics/difference-between-informed-and-uninformed-search/"
    },
    {
        "id": "scipub_agent_search_paper_2021",
        "type": "pdf",
        "publisher": "Science Publications",
        "url": "https://thescipub.com/pdf/jcssp.2021.1147.1156.pdf"
    },
    {
        "id": "ibm_ai_agent_orchestration",
        "type": "html",
        "publisher": "IBM",
        "url": "https://www.ibm.com/think/topics/ai-agent-orchestration"
    },
    {
        "id": "domo_ai_agent_orchestration",
        "type": "html",
        "publisher": "Domo",
        "url": "https://www.domo.com/glossary/ai-agent-orchestration"
    },
    {
        "id": "aimultiple_agentic_frameworks",
        "type": "html",
        "publisher": "AI Multiple",
        "url": "https://research.aimultiple.com/agentic-frameworks/"
    },
    {
        "id": "reddit_multiagent_system_evaluator",
        "type": "html",
        "publisher": "Reddit",
        "url": "https://www.reddit.com/r/PromptSynergy/comments/1np7wxw/multiagent_system_evaluator_with_40point_analysis/"
    },
    {
        "id": "dextra_ai_agent_orchestration",
        "type": "html",
        "publisher": "Dextra Labs",
        "url": "https://dextralabs.com/blog/what-is-ai-agent-orchestration/"
    },
    {
        "id": "kubiya_agent_orchestration_frameworks",
        "type": "html",
        "publisher": "Kubiya",
        "url": "https://www.kubiya.ai/blog/ai-agent-orchestration-frameworks"
    },
    {
        "id": "projectpro_ai_agent_evaluation",
        "type": "html",
        "publisher": "ProjectPro",
        "url": "https://www.projectpro.io/article/ai-agent-evaluation/1178"
    },
    {
        "id": "zyrix_multi_agent_testing_guide_2025",
        "type": "html",
        "publisher": "Zyrix AI",
        "url": "https://zyrix.ai/blogs/multi-agent-ai-testing-guide-2025/"
    }
]

# -----------------------------
# Utilities
# -----------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


STOPWORDS = {
    "a","an","and","are","as","at","be","but","by","can","do","does","for","from","how","i","if","in","is","it","of","on","or",
    "that","the","their","then","there","these","this","to","was","were","what","when","where","which","who","why","with","you","your"
}

def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def extract_tags(text: str, title: Optional[str], max_tags: int = 8) -> List[str]:
    content = " ".join([t for t in [title, text] if t])
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

def normalize_url(url: str) -> str:
    if url.endswith("title="):
        return url[:-6].rstrip("?&")
    return url

def extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = soup.get_text("\n")
    return clean_ws(text)


def safe_get(session: requests.Session, url: str) -> requests.Response:
    # basic retry for transient blocks
    last_exc = None
    for attempt in range(3):
        try:
            r = session.get(url, timeout=TIMEOUT_S, allow_redirects=True)
            return r
        except Exception as e:
            last_exc = e
            time.sleep(1.25 * (attempt + 1))
    raise last_exc


# -----------------------------
# Metadata extraction (best effort)
# -----------------------------
def extract_meta_from_html(html: str, url: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Returns: (title, author, publication_date_iso)
    Best-effort using meta tags commonly found in blogs/news sites.
    """
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)

    # Common meta tags
    def meta(name: str) -> Optional[str]:
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return tag["content"].strip()
        tag = soup.find("meta", attrs={"property": name})
        if tag and tag.get("content"):
            return tag["content"].strip()
        return None

    title2 = meta("og:title") or meta("twitter:title")
    if title2:
        title = title2

    author = meta("author") or meta("article:author") or meta("og:article:author")
    pub = meta("article:published_time") or meta("og:article:published_time") or meta("pubdate") or meta("date")

    # Normalize date to ISO if possible (keep as-is if parsing fails)
    pub_iso = None
    if pub:
        # Many sites already provide ISO; keep it if it looks like ISO
        if re.match(r"^\d{4}-\d{2}-\d{2}", pub):
            pub_iso = pub
        else:
            # Try minimal parsing like "Jan 10, 2025"
            try:
                from dateutil import parser as dtparser  # python-dateutil in requirements
                pub_iso = dtparser.parse(pub).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            except Exception:
                pub_iso = pub  # best-effort fallback

    return title.strip(), (author.strip() if author else None), (pub_iso.strip() if pub_iso else None)


# -----------------------------
# HTML extraction
# -----------------------------
def extract_main_text_readability(html: str) -> Tuple[str, str]:
    doc = Document(html)
    title = doc.short_title() or ""
    summary_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(summary_html, "html.parser")

    parts = []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        t = el.get_text(" ", strip=True)
        if t:
            parts.append(t)
    text = "\n".join(parts)
    return title.strip(), clean_ws(text)


def fetch_html_article(session: requests.Session, url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    url = normalize_url(url)
    r = safe_get(session, url)
    if r.status_code == 403:
        return None, None, None, f"403 Forbidden (site blocked requests): {url}"
    if r.status_code >= 400:
        return None, None, None, f"HTTP {r.status_code}: {url}"

    html = r.text
    meta_title, author, pub_date = extract_meta_from_html(html, url)
    title, text = extract_main_text_readability(html)

    # Prefer readability title but fall back to meta
    final_title = title or meta_title or url

    # Fallback if readability is too thin
    if not text or len(text) < 500:
        soup = BeautifulSoup(html, "html.parser")
        raw = "\n".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        raw = clean_ws(raw)
        if len(raw) > len(text):
            text = raw
    if not text or len(text) < 300:
        raw = extract_visible_text(html)
        if len(raw) > len(text or ""):
            text = raw

    if not text or len(text) < 200:
        return None, None, None, f"Could not extract sufficient text from: {url}"

    return final_title, author, pub_date, text


# -----------------------------
# PDF extraction (arXiv etc.)
# -----------------------------
def fetch_pdf_text(session: requests.Session, url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    r = safe_get(session, url)
    if r.status_code >= 400:
        return None, None, None, f"HTTP {r.status_code}: {url}"

    # Save temp pdf
    os.makedirs(os.path.join(OUT_DIR, "_tmp"), exist_ok=True)
    tmp_path = os.path.join(OUT_DIR, "_tmp", f"tmp_{int(time.time()*1000)}.pdf")
    with open(tmp_path, "wb") as f:
        f.write(r.content)

    # Extract text
    try:
        text = pdfminer_extract_text(tmp_path) or ""
    finally:
        # remove tmp
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    text = clean_ws(text)
    if not text or len(text) < 800:
        return None, None, None, f"PDF text extraction too small for: {url}"

    # Title/author/date for arXiv PDFs: best-effort from first page text
    # Keep these optional; you can enrich later via arXiv API if you want.
    title = "arXiv paper"
    author = None
    pub_date = None
    return title, author, pub_date, text


# -----------------------------
# Main ingestion
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    written = 0
    skipped = []
    manifest_docs = []

    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for src in SOURCES:
            doc_id = f"article::{src['id']}"
            url = src["url"]
            publisher = src.get("publisher")

            if src["type"] == "html":
                title, author, pub_date, text_or_err = fetch_html_article(session, url)
            elif src["type"] == "pdf":
                title, author, pub_date, text_or_err = fetch_pdf_text(session, url)
            else:
                skipped.append({"id": src["id"], "url": url, "reason": f"Unknown type: {src['type']}"})
                continue

            if title is None:
                skipped.append({"id": src["id"], "url": url, "reason": text_or_err})
                continue

            text = text_or_err
            chunks = chunk_text(text, size=1200, overlap=150)
            if not chunks:
                skipped.append({"id": src["id"], "url": url, "reason": "No chunks produced"})
                continue

            for i, chunk in enumerate(chunks, 1):
                breadcrumbs = f"Article: {title}"
                tags = extract_tags(chunk, title)
                rec = {
                    "chunk_id": f"{doc_id}::{i:06d}",
                    "doc_id": doc_id,
                    "doc_title": title,
                    "title": title,
                    "doc_type": "article",
                    "publisher": publisher,
                    "author": author,
                    "publication_date": pub_date,
                    "source_url": url,
                    "section_title": None,
                    "page_start": None,
                    "page_end": None,
                    "source_type": "article",
                    "date": pub_date,
                    "url": url,
                    "priority": 1,
                    "tags": tags,
                    "breadcrumbs": breadcrumbs,
                    "chunk_type": "section",
                    "text": f"Breadcrumbs: {breadcrumbs}\n{chunk}",
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

            manifest_docs.append(
                {
                    "id": doc_id,
                    "title": title,
                    "format": "pdf" if src["type"] == "pdf" else "html",
                    "filename": url,
                    "blocks": len(chunks),
                    "source_type": "article",
                    "url": url,
                    "publisher": publisher,
                    "author": author,
                    "publication_date": pub_date,
                    "date": pub_date,
                }
            )
            print(f"[OK] {src['id']}: {len(chunks)} chunks")

    manifest = {
        "generated_at": now_iso(),
        "documents": manifest_docs,
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Write a small ingestion report
    report_path = os.path.join(OUT_DIR, "articles_ingest_report.json")
    report = {
        "generated_at": now_iso(),
        "out_jsonl": OUT_JSONL,
        "out_manifest": OUT_MANIFEST,
        "total_chunks_written": written,
        "sources_total": len(SOURCES),
        "sources_skipped": skipped,
        "notes": [
            "Towards Data Science links may return 403 and are skipped to keep the pipeline reproducible.",
            "arXiv PDFs are ingested via pdfminer; title/author/date may be enriched later.",
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Wrote {written} chunks to {OUT_JSONL}")
    if skipped:
        print(f"[WARN] Skipped {len(skipped)} sources. See {report_path}.")


if __name__ == "__main__":
    main()
