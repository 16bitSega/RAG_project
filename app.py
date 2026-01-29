import os
import re
import json
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from statistics import median
from datetime import datetime, timezone

from dotenv import load_dotenv
import streamlit as st
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

COMPANY_NAME = "O_O.inc"
COMPANY_EMAIL = "o.obolonsky@proton.me"
COMPANY_PHONE = "+380953555919"
COMPANY_ABOUT = "AI Software development company ready to collaborate and make your ideas come true"

REPO_OWNER = "16bitSega"
REPO_NAME = "RAG_project"

BOOK_CHUNKS_PATH = os.environ.get("RAG_BOOK_CHUNKS_PATH", "data/normalized/chunks_books.jsonl")
ARTICLE_CHUNKS_PATH = os.environ.get("RAG_ARTICLE_CHUNKS_PATH", "data/normalized/chunks_articles.jsonl")
BOOK_MANIFEST_PATH = os.environ.get("RAG_BOOK_MANIFEST_PATH", "data/normalized/manifest_books.json")
ARTICLE_MANIFEST_PATH = os.environ.get("RAG_ARTICLE_MANIFEST_PATH", "data/normalized/manifest_articles.json")
BOOK_INDEX_PATH = os.environ.get("RAG_BOOK_INDEX_PATH", "data/normalized/index_books.faiss")
ARTICLE_INDEX_PATH = os.environ.get("RAG_ARTICLE_INDEX_PATH", "data/normalized/index_articles.faiss")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

OLLAMA_BASE_URL = os.environ.get("RAG_OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("RAG_OLLAMA_MODEL", "llama3.2:1b")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
# Retrieval mix: book-first + article nuance.
BOOK_K = 12
ARTICLE_K = 4
PER_DOC_CAP = 3
OVERLAP_FILTER = True

# Enhanced answer mix: heavier retrieval for deeper answers.
ENHANCED_BOOK_K = 12
ENHANCED_ARTICLE_K = 5

AVOID_PHRASES = [
    "The article discusses",
    "The article presents",
    "The authors propose",
    "Overall, the article",
    "This paper",
    "This study",
    "The paper",
]

SOCIAL_TERMS = [
    "LinkedIn",
    "Reddit",
    "Twitter",
    "X",
    "Facebook",
    "Instagram",
    "TikTok",
    "YouTube",
]


ARTICLE_PREFIX = "article::"

STOPWORDS = {
    "a","an","and","are","as","at","be","but","by","can","do","does","for","from","how","i","if","in","is","it","of","on","or",
    "that","the","their","then","there","these","this","to","was","were","what","when","where","which","who","why","with","you","your"
}

AIMA_QUESTIONS = [
    "In what ways do knowledge representation choices limit or enable reasoning?",
    "What distinguishes rational agents from intelligent behavior in practice?",
]

AGENTIC_QUESTIONS = [
    "How should agents balance planning with reactive decision-making?",
    "What role does memory play in enabling long-horizon agent behavior?",
]

GENAI_QUESTIONS = [
    "What recurring failure patterns appear when deploying generative AI systems in production?",
    "How can an agent manage memory (short-term vs long-term) without leaking sensitive data?",
]

ARTICLE_QUESTIONS_DEFAULT = [
    "How does Model Context Protocol (MCP) help orchestrate AI agents and tools?",
    "What is the core idea behind retrieval-augmented generation (RAG)?",
    "How to build an orchestration agent system?",
]

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_title: Optional[str] = None
    source_url: Optional[str] = None
    published_at: Optional[str] = None
    author: Optional[str] = None
    doc_type: Optional[str] = None

def safe_text(s: str) -> str:
    return html.escape(s or "")

def normalize_display_text(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if len(lines) >= 12:
        short = sum(1 for ln in lines if len(ln.split()) <= 2)
        if short / max(1, len(lines)) >= 0.7:
            s = " ".join(lines)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_company_question(q: str) -> bool:
    q = (q or "").lower()
    patterns = [
        r"where are you working",
        r"where do you work",
        r"who do you work for",
        r"company (name|info|details)",
        r"contact (info|details|email|phone)",
        r"your email",
        r"your phone",
        r"about your company",
    ]
    return any(re.search(p, q) for p in patterns)

def company_answer() -> str:
    return (
        f"Company: {COMPANY_NAME}\n"
        f"Email: {COMPANY_EMAIL}\n"
        f"Phone: {COMPANY_PHONE}\n"
        f"About: {COMPANY_ABOUT}"
    )

def sanitize_answer(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"https?://\S+", "", text)
    cleaned = re.sub(r"\bwww\.\S+", "", cleaned)
    for phrase in AVOID_PHRASES:
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b[:,]*", "", cleaned, flags=re.IGNORECASE)
    for term in SOCIAL_TERMS:
        cleaned = re.sub(rf"\b{re.escape(term)}\b", "", cleaned)
    cleaned = re.sub(r"(?mi)^sources:?\s*$", "", cleaned)
    cleaned = re.sub(r"(?mi)^\[[AB]\d+\].*$", "", cleaned)
    cleaned = re.sub(r"(?i)\bnot found in dataset\b\.?", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()




def read_manifest(path: str) -> Dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {"documents": []}

def read_chunks_jsonl(path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=obj.get("chunk_id", ""),
                    doc_id=obj.get("doc_id", ""),
                    text=obj.get("text", "") or "",
                    page_start=obj.get("page_start"),
                    page_end=obj.get("page_end"),
                    section_title=obj.get("section_title"),
                    source_url=obj.get("source_url"),
                    published_at=obj.get("published_at"),
                    author=obj.get("author"),
                    doc_type=obj.get("doc_type"),
                )
            )
    return chunks

def slug_to_title(slug: str) -> str:
    slug = (slug or "").replace("_", " ").replace("-", " ").strip()
    return " ".join(w.capitalize() for w in slug.split())

def infer_source_type(doc_id: str, meta: Optional[Dict] = None) -> str:
    if meta and meta.get("source_type"):
        return str(meta["source_type"])
    if doc_id.startswith("mcp::"):
        return "mcp"
    if doc_id.startswith("article::"):
        return "article"
    return "book"

def build_doc_index(manifest: Dict) -> Dict[str, Dict]:
    by_id: Dict[str, Dict] = {}
    for d in manifest.get("documents", []) or []:
        doc_id = str(d.get("id") or "")
        if not doc_id:
            continue
        meta = dict(d)
        meta.setdefault("source_type", infer_source_type(doc_id, meta))
        by_id[doc_id] = meta
    return by_id

def merge_doc_indexes(*indexes: Dict[str, Dict]) -> Dict[str, Dict]:
    merged: Dict[str, Dict] = {}
    for idx in indexes:
        for doc_id, meta in idx.items():
            if doc_id not in merged:
                merged[doc_id] = meta
    return merged


def compute_stats(chunks: List[Chunk], manifest: Dict, doc_index: Dict[str, Dict]) -> Dict:
    lengths = [len(c.text) for c in chunks if c.text]
    with_pages = sum(1 for c in chunks if c.page_start is not None)
    with_sections = sum(1 for c in chunks if c.section_title)

    doc_counts = Counter(c.doc_id for c in chunks)
    type_counts = Counter(infer_source_type(c.doc_id, doc_index.get(c.doc_id)) for c in chunks)

    docs = manifest.get("documents", []) or []
    mcp_docs = [d for d in docs if str(d.get("id", "")).startswith("mcp::")]
    other_docs = [d for d in docs if not str(d.get("id", "")).startswith("mcp::")]

    mcp_blocks = sum(int(d.get("blocks", 0) or 0) for d in mcp_docs)

    def fmt_doc(d: Dict) -> str:
        title = d.get("title") or d.get("id") or "unknown"
        doc_id = d.get("id") or ""
        fmt = d.get("format") or ""
        blocks = d.get("blocks", None)
        bits = [f"{title} ({doc_id})"]
        if fmt:
            bits.append(fmt)
        if blocks is not None:
            bits.append(f"{blocks} blocks")
        return " · ".join(bits)

    sources_lines = [fmt_doc(d) for d in other_docs]

    return {
        "total_chunks": len(chunks),
        "length_min": min(lengths) if lengths else 0,
        "length_median": int(median(lengths)) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "with_pages": with_pages,
        "with_sections": with_sections,
        "type_counts": dict(type_counts),
        "mcp_docs_count": len(mcp_docs),
        "mcp_blocks_total": mcp_blocks,
        "sources_lines": sources_lines,
    }
@st.cache_resource(show_spinner=False)
def load_embedder(_model_name: str) -> SentenceTransformer:
    return SentenceTransformer(_model_name)

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index

def load_or_build_index(
    chunks: List[Chunk],
    embedder: SentenceTransformer,
    index_path: str,
    source_path: Optional[str] = None,
) -> faiss.Index:
    p = Path(index_path)
    src = Path(source_path) if source_path else None
    if p.exists() and (not src or not src.exists() or p.stat().st_mtime >= src.stat().st_mtime):
        return faiss.read_index(str(p))
    texts = [c.text for c in chunks]
    vecs = embedder.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype="float32")
    index = build_faiss_index(vecs)
    p.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(p))
    return index

def retrieve(query: str, embedder: SentenceTransformer, index: faiss.Index, chunks: List[Chunk], k: int = 8) -> List[Tuple[float, Chunk]]:
    qv = embedder.encode([query], normalize_embeddings=True)
    qv = np.asarray(qv, dtype="float32")
    D, I = index.search(qv, k)
    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(chunks):
            continue
        hits.append((float(score), chunks[idx]))
    return hits

def extract_keywords(q: str) -> List[str]:
    toks = re.findall(r"[a-zA-Z0-9_]+", (q or "").lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t) >= 3]
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out[:10]

def not_found_by_terms(question: str, hits: List[Tuple[float, Chunk]]) -> bool:
    terms = extract_keywords(question)
    if not terms:
        return False
    blob = " ".join((c.text or "").lower() for _, c in hits)
    return not any(t in blob for t in terms)

def build_citation_tags(hits: List[Tuple[float, Chunk]], doc_index: Dict[str, Dict]) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    counts = {"book": 0, "article": 0}
    for _, c in hits:
        if c.doc_id in tags:
            continue
        source_type = infer_source_type(c.doc_id, doc_index.get(c.doc_id))
        if source_type == "article":
            counts["article"] += 1
            tags[c.doc_id] = f"[A{counts['article']}]"
        else:
            counts["book"] += 1
            tags[c.doc_id] = f"[B{counts['book']}]"
    return tags

def format_citation(c: Chunk, doc_index: Dict[str, Dict], tags: Dict[str, str]) -> str:
    meta = doc_index.get(c.doc_id, {})
    title = meta.get("title") or slug_to_title(c.doc_id.replace(ARTICLE_PREFIX, ""))
    tag = tags.get(c.doc_id, "[B?]")
    if c.page_start is not None:
        return f"{tag} {title} p.{c.page_start}"
    return f"{tag} {title}"

def chunk_heading(c: Chunk, doc_index: Dict[str, Dict], tags: Dict[str, str]) -> str:
    base = format_citation(c, doc_index, tags)
    section = c.section_title or ""
    if section:
        return f"{base} - {section}"
    return base

def build_context(
    book_hits: List[Tuple[float, Chunk]],
    article_hits: List[Tuple[float, Chunk]],
    doc_index: Dict[str, Dict],
    tags: Dict[str, str],
    max_chars_per_chunk: int = 1400,
) -> str:
    book_parts = []
    article_parts = []
    for _, c in book_hits:
        t = normalize_display_text(c.text)
        if len(t) > max_chars_per_chunk:
            t = t[:max_chars_per_chunk] + "..."
        heading = chunk_heading(c, doc_index, tags)
        book_parts.append(f"{heading}\n{t}")

    for _, c in article_hits:
        t = normalize_display_text(c.text)
        if len(t) > max_chars_per_chunk:
            t = t[:max_chars_per_chunk] + "..."
        heading = chunk_heading(c, doc_index, tags)
        article_parts.append(f"{heading}\n{t}")

    parts = []
    if book_parts:
        parts.append("BOOK EXCERPTS:\n" + "\n\n".join(book_parts))
    if article_parts:
        parts.append("ARTICLE EXCERPTS:\n" + "\n\n".join(article_parts))
    return "\n\n".join(parts)

def chunk_keyword_overlap(chunk: Chunk, terms: List[str]) -> int:
    if not terms:
        return 0
    text = (chunk.text or "").lower()
    return sum(1 for t in terms if t in text)

def limit_by_doc(hits: List[Tuple[float, Chunk]], cap: int) -> List[Tuple[float, Chunk]]:
    if cap <= 0:
        return hits
    counts: Dict[str, int] = {}
    out: List[Tuple[float, Chunk]] = []
    for score, chunk in hits:
        cnt = counts.get(chunk.doc_id, 0)
        if cnt >= cap:
            continue
        counts[chunk.doc_id] = cnt + 1
        out.append((score, chunk))
    return out

def refine_hits(hits: List[Tuple[float, Chunk]], query: str) -> List[Tuple[float, Chunk]]:
    terms = extract_keywords(query)
    scored = []
    for score, chunk in hits:
        overlap = chunk_keyword_overlap(chunk, terms)
        scored.append((overlap, score, chunk))
    if OVERLAP_FILTER and scored and max(s[0] for s in scored) > 0:
        scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [(score, chunk) for _, score, chunk in scored]

def retrieve_books_and_articles(
    query: str,
    embedder: SentenceTransformer,
    book_index: faiss.Index,
    book_chunks: List[Chunk],
    article_index: faiss.Index,
    article_chunks: List[Chunk],
    book_k: int,
    article_k: int,
) -> Tuple[List[Tuple[float, Chunk]], List[Tuple[float, Chunk]]]:
    oversample_book = book_k * 2
    oversample_article = article_k * 2
    book_hits = retrieve(query, embedder, book_index, book_chunks, k=oversample_book)
    article_hits = retrieve(query, embedder, article_index, article_chunks, k=oversample_article)
    book_hits = refine_hits(book_hits, query)
    article_hits = refine_hits(article_hits, query)
    book_hits = limit_by_doc(book_hits, PER_DOC_CAP)[:book_k]
    article_hits = limit_by_doc(article_hits, PER_DOC_CAP)[:article_k]
    return book_hits, article_hits

def answer_question(
    question: str,
    *,
    book_k: int,
    article_k: int,
    enhanced: bool = False,
) -> Tuple[str, List[str], bool]:
    book_hits, article_hits = retrieve_books_and_articles(
        question,
        embedder,
        book_index,
        book_chunks,
        article_index,
        article_chunks,
        book_k,
        article_k,
    )
    all_hits = book_hits + article_hits
    citation_tags = build_citation_tags(all_hits, doc_index)
    citations = [format_citation(c, doc_index, citation_tags) for _, c in all_hits]
    if not all_hits or not_found_by_terms(question, all_hits):
        return "Not found in dataset.", citations, False

    context = build_context(book_hits, article_hits, doc_index, citation_tags)
    avoid_text = "; ".join(AVOID_PHRASES)
    base_rules = (
        "You must answer using only the provided context.\n"
        "Use BOOK excerpts for core claims; use ARTICLE excerpts only for nuance or examples.\n"
        "Cite sources inline using the provided tags.\n"
        f"Avoid boilerplate phrases such as: {avoid_text}.\n"
        "Do not include social network names or links in the answer.\n"
        "If the context does not contain the answer, output exactly: Not found in dataset.\n"
    )
    if enhanced:
        format_rules = (
            "You must answer using only the provided context.\n"
            "Use BOOK excerpts for core claims; use ARTICLE excerpts only for nuance or examples.\n"
            "Cite sources inline using the provided tags.\n"
            f"Avoid boilerplate phrases such as: {avoid_text}.\n"
            "Do not include social network names or links in the answer.\n"
            "Answer must explicitly restate the user's question in the opening sentence.\n"
            "Then provide a deeper synthesis that integrates multiple sources.\n"
            "Write as one comprehensive narrative, not a list or outline.\n"
            "Avoid meta-prefaces like \"The provided text appears\" or \"The excerpts discuss\".\n"
            "When presenting alternative views, use the phrase \"In other perspective...\" to separate them.\n"
        )
    else:
        format_rules = "Answer the question directly and succinctly. No self-reference.\n"
    prompt = (
        base_rules
        + format_rules
        + f"\nQuestion:\n{question}\n\nContext:\n{context}\n\nAnswer:"
    )
    answer, err = ollama_chat(prompt)
    if err or not answer:
        return "Not found in dataset.", citations, False
    return sanitize_answer(answer), citations, True

def ollama_chat(prompt: str, timeout: Tuple[int,int] = (10, 600)) -> Tuple[str, Optional[str]]:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": f"You are an assistant for {COMPANY_NAME}. Contact: {COMPANY_EMAIL}, {COMPANY_PHONE}. {COMPANY_ABOUT}. Answer only from the provided context. Keep answers concise. Cite sources using the provided citation tags exactly."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2}
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        msg = (data.get("message") or {}).get("content") or ""
        return msg.strip(), None
    except Exception as e:
        return "", str(e)

def github_create_issue(title: str, body: str, labels: Optional[List[str]] = None) -> Tuple[Optional[int], Optional[str]]:
    if not GITHUB_TOKEN:
        return None, "Missing GITHUB_TOKEN"
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    payload = {"title": title, "body": body}
    if labels:
        payload["labels"] = labels
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=(10, 60))
        r.raise_for_status()
        j = r.json()
        return int(j.get("number")), None
    except Exception as e:
        return None, str(e)

def valid_email(email: str) -> bool:
    email = (email or "").strip()
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email))

def build_issue_body(user_name: str, user_email: str, summary: str, description: str, question: str, citations: List[str]) -> str:
    now = datetime.now(timezone.utc).isoformat()
    cits = "\n".join(f"- {c}" for c in citations) if citations else "(none)"
    return (
        f"Reporter: {user_name} <{user_email}>\n"
        f"Time (UTC): {now}\n"
        f"Company: {COMPANY_NAME}\n"
        f"Contact: {COMPANY_EMAIL} / {COMPANY_PHONE}\n\n"
        f"Summary:\n{summary}\n\n"
        f"Description:\n{description}\n\n"
        f"User question:\n{question}\n\n"
        f"Evidence (citations):\n{cits}\n"
    )

st.set_page_config(page_title="Agentic RAG", layout="wide")

st.markdown(
    """
<style>
.sticky-wrap{position:sticky;top:0;z-index:50;background:rgba(14,17,23,0.98);padding:0.75rem 0.75rem 0.25rem 0.75rem;border-bottom:1px solid rgba(255,255,255,0.08);}
.stacked-control{display:block;}
.stacked-control .stButton button{border-radius:10px !important;}
.sources-btn .stButton button{background:#2f6b3f !important;color:#111 !important;border:1px solid #8bd17c !important;}
.sources-btn button{background:#2f6b3f !important;color:#111 !important;border:1px solid #8bd17c !important;}
button[aria-label^="MCP •"]{position:relative;padding-left:2.2rem;}
button[aria-label^="MCP •"]::before{content:"MCP";position:absolute;left:0.6rem;top:50%;transform:translateY(-50%);background:#2a3b4f;color:#f5e6b8;border:2px solid #c9a227;border-radius:8px;padding:0.05rem 0.35rem;font-weight:700;letter-spacing:0.04em;box-shadow:inset 0 0 0 2px rgba(0,0,0,0.25);}
</style>
""",
    unsafe_allow_html=True,
)

if "is_thinking" not in st.session_state:
    st.session_state["is_thinking"] = False

with st.sidebar:
    st.markdown(f"**Company:** {COMPANY_NAME}")
    st.markdown(f"**Contact:** {COMPANY_EMAIL} · {COMPANY_PHONE}")
    st.caption(COMPANY_ABOUT)
    st.write("")
    st.subheader("Support")
    st.caption("If an answer is not found in the dataset, you can create a support ticket (GitHub issue).")
    st.session_state.setdefault("open_ticket_ui", False)
    if st.button("Open ticket form", use_container_width=True, disabled=st.session_state["is_thinking"]):
        st.session_state["open_ticket_ui"] = True
    st.write("")
    st.subheader("LLM")
    st.markdown(f"- Model: `{OLLAMA_MODEL}`")
    st.markdown(f"- URL: `{OLLAMA_BASE_URL}`")
    st.write("")
    st.subheader("Embedding model (retrieval)")
    st.code(EMBED_MODEL)
    st.write("")
    st.subheader("Retrieval settings")
    st.caption(f"book_k={BOOK_K}, article_k={ARTICLE_K}, per_doc_cap={PER_DOC_CAP}, overlap_filter={OVERLAP_FILTER}")
    st.subheader("Dataset stats")
    st.caption("Local dataset only")
@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> List[Chunk]:
    return read_chunks_jsonl(path)

book_chunks = load_dataset(BOOK_CHUNKS_PATH)
article_chunks = load_dataset(ARTICLE_CHUNKS_PATH)
book_manifest = read_manifest(BOOK_MANIFEST_PATH)
article_manifest = read_manifest(ARTICLE_MANIFEST_PATH)
book_doc_index = build_doc_index(book_manifest)
article_doc_index = build_doc_index(article_manifest)
doc_index = merge_doc_indexes(book_doc_index, article_doc_index)
book_stats = compute_stats(book_chunks, book_manifest, book_doc_index)
article_stats = compute_stats(article_chunks, article_manifest, article_doc_index)
embedder = load_embedder(EMBED_MODEL)
book_index = load_or_build_index(book_chunks, embedder, BOOK_INDEX_PATH, BOOK_CHUNKS_PATH)
article_index = load_or_build_index(article_chunks, embedder, ARTICLE_INDEX_PATH, ARTICLE_CHUNKS_PATH)

if "chat" not in st.session_state:
    st.session_state["chat"] = []
if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = ""
if "ticket_prefill" not in st.session_state:
    st.session_state["ticket_prefill"] = None
if "enhancing_key" not in st.session_state:
    st.session_state["enhancing_key"] = None
if "active_action" not in st.session_state:
    st.session_state["active_action"] = None

def push_message(role: str, content: str, citations: Optional[List[str]] = None, not_found: bool = False):
    msg = {"role": role, "content": content, "ts": datetime.now().isoformat()}
    if citations:
        msg["citations"] = citations
    if not_found:
        msg["not_found"] = True
    st.session_state["chat"].append(msg)

def sample_click(q: str):
    st.session_state["pending_question"] = q

def start_action(action_type: str, payload: Dict):
    if st.session_state["is_thinking"] or st.session_state.get("active_action"):
        return
    st.session_state["is_thinking"] = True
    st.session_state["active_action"] = {"type": action_type, "payload": payload}
    st.rerun()

def parse_generated_questions(text: str) -> List[str]:
    lines = [ln.strip(" -\t") for ln in (text or "").splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        ln = re.sub(r"^\d+[\).]\s*", "", ln).strip()
        if ln and ln not in cleaned:
            cleaned.append(ln)
        if len(cleaned) >= 3:
            break
    return cleaned

with st.sidebar:
    st.write("")
    st.markdown("**Books + MCP**")
    st.write(f"Chunk length: min {book_stats['length_min']}, median {book_stats['length_median']}, max {book_stats['length_max']}")
    st.write("")
    st.markdown("**Articles**")
    st.write(f"Chunk length: min {article_stats['length_min']}, median {article_stats['length_median']}, max {article_stats['length_max']}")
    st.write("")
    st.markdown("**By type (inferred)**")
    for k in ["book", "mcp", "article"]:
        total = 0
        if k in book_stats["type_counts"]:
            total += book_stats["type_counts"][k]
        if k in article_stats["type_counts"]:
            total += article_stats["type_counts"][k]
        if total:
            st.write(f"{k}: {total}")
    st.write("")
    st.session_state.setdefault("show_sources", False)
    st.markdown('<div class="stacked-control sources-btn">', unsafe_allow_html=True)
    if st.button("Sources (click to expand the list)", use_container_width=True, disabled=st.session_state["is_thinking"]):
        st.session_state["show_sources"] = not st.session_state["show_sources"]
    st.markdown("</div>", unsafe_allow_html=True)
    if st.session_state["show_sources"]:
        if book_stats["mcp_docs_count"]:
            mcp_line = f"MCP: {book_stats['mcp_docs_count']} docs"
            if book_stats["mcp_blocks_total"]:
                mcp_line += f", {book_stats['mcp_blocks_total']} blocks"
            st.write(mcp_line)
        for line in book_stats["sources_lines"]:
            st.write(line)
        if article_stats["sources_lines"]:
            st.write("")
            st.markdown("**Article sources**")
            for line in article_stats["sources_lines"]:
                st.write(line)

def run_enhance(question: str, enhanced_key: str):
    if not question or not enhanced_key:
        return
    st.session_state["enhancing_key"] = enhanced_key
    answer, citations, ok = answer_question(
        question,
        book_k=ENHANCED_BOOK_K,
        article_k=ENHANCED_ARTICLE_K,
        enhanced=True,
    )
    if ok:
        st.session_state[enhanced_key] = {"answer": answer, "citations": citations, "not_found": False}
    else:
        st.session_state[enhanced_key] = {"answer": "Not found in dataset.", "citations": [], "not_found": True}
        st.session_state["ticket_prefill"] = {"question": question, "citations": citations}
    st.session_state["enhancing_key"] = None

def run_regen():
    gen_prompt = (
        "Generate exactly 3 concise user questions about MCP and AI agents orchestration. "
        "Return each question on its own line without extra text."
    )
    text, err = ollama_chat(gen_prompt)
    if err or not text:
        st.warning(f"LLM request failed: {err}")
        return
    qs = parse_generated_questions(text)
    if len(qs) == 3:
        st.session_state["article_questions"] = qs
    else:
        st.warning("Could not parse 3 questions. Try again.")

left, right = st.columns([3, 1], vertical_alignment="top")

with right:
    st.markdown("### Chat history")
    qa_pairs = []
    for i in range(len(st.session_state["chat"]) - 1):
        if st.session_state["chat"][i]["role"] == "user" and st.session_state["chat"][i+1]["role"] == "assistant":
            qa_pairs.append(
                {
                    "question": st.session_state["chat"][i]["content"],
                    "index": i,
                    "ts": st.session_state["chat"][i].get("ts"),
                }
            )
    qa_pairs.sort(key=lambda x: x["ts"] or "", reverse=True)
    labels = ["—"]
    for item in qa_pairs:
        q = item["question"]
        ts = item["ts"]
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                t_label = dt.strftime("%H:%M")
            except ValueError:
                t_label = "--:--"
        else:
            t_label = "--:--"
        q_label = q if len(q) <= 60 else q[:57] + "..."
        labels.append(f"{t_label} — {q_label}")
    sel = st.selectbox("Recent questions", options=labels, index=0)
    if sel == "—":
        st.session_state["selected_pair_index"] = None
    else:
        idx = labels.index(sel) - 1
        st.session_state["selected_pair_index"] = qa_pairs[idx]["index"]
with left:
    st.markdown('<div class="sticky-wrap">', unsafe_allow_html=True)
    st.markdown("## Agentic RAG: Books + MCP + Articles")
    st.caption("You are welcome to ask your question related to AI Agents in the chat window or tap on sample questions below. The application responds with citations (file and page when available) and chunk IDs. Sources are listed in Dataset stats in the sidebar.")
    question = st.chat_input("Ask a question (dataset-only)", disabled=st.session_state["is_thinking"])
    if (st.session_state.get("active_action") or {}).get("type") == "answer":
        st.markdown("**Thinking...**")
        st.spinner("Thinking...")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("pending_question"):
        if not question:
            question = st.session_state["pending_question"]
        st.session_state["pending_question"] = ""

    if question:
        q_norm = question.strip()
        if q_norm.lower() == "create support ticket":
            st.session_state["open_ticket_ui"] = True
            st.session_state["ticket_prefill"] = {"question": "", "citations": []}
        else:
            start_action("answer", {"question": q_norm})
            st.stop()
            st.rerun()

    sel_i = st.session_state.get("selected_pair_index")
    if sel_i is None:
        # show last Q/A pair
        sel_i = None
        for j in range(len(st.session_state["chat"]) - 2, -1, -1):
            if st.session_state["chat"][j]["role"] == "user" and j + 1 < len(st.session_state["chat"]) and st.session_state["chat"][j+1]["role"] == "assistant":
                sel_i = j
                break

    if sel_i is not None:
        qmsg = st.session_state["chat"][sel_i]
        amsg = st.session_state["chat"][sel_i + 1]
        st.markdown(f"**Question:** {safe_text(qmsg['content'])}", unsafe_allow_html=True)
        st.markdown(f"**Answer:** {safe_text(amsg['content'])}", unsafe_allow_html=True)
        if amsg.get("citations") and not amsg.get("not_found"):
            show_key = f"show_sources_answer_{sel_i}"
            st.session_state.setdefault(show_key, False)
            st.markdown('<div class="stacked-control sources-btn">', unsafe_allow_html=True)
            if st.button(
                "Sources (click to expand the list)",
                key=f"sources_btn_{sel_i}",
                use_container_width=True,
                disabled=st.session_state["is_thinking"],
            ):
                st.session_state[show_key] = not st.session_state[show_key]
            st.markdown("</div>", unsafe_allow_html=True)
            if st.session_state[show_key]:
                for c in amsg["citations"]:
                    st.markdown(f"- {safe_text(c)}", unsafe_allow_html=True)
        if amsg.get("not_found"):
            st.info("Not found in dataset. If you believe the topic is missing, please open a support ticket.")
            if st.button("Open ticket form", key="ticket_btn_single", use_container_width=False, disabled=st.session_state["is_thinking"]):
                st.session_state["open_ticket_ui"] = True
        else:
            enhanced_key = f"enhanced_answer_{sel_i}"
            enhance_btn_key = f"enhance_btn_{sel_i}"
            ticket_key = f"ticket_btn_{sel_i}"
            show_enhance_ui = enhanced_key not in st.session_state and st.session_state.get("enhancing_key") != enhanced_key
            if show_enhance_ui:
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    if st.button(
                        "Enhance the answer (x2 chunks)",
                        key=enhance_btn_key,
                        use_container_width=True,
                        disabled=st.session_state["is_thinking"],
                    ):
                        q_text = qmsg.get("content", "") or st.session_state.get("last_question") or ""
                        if q_text:
                            start_action("enhance", {"question": q_text, "enhanced_key": enhanced_key})
                            st.stop()
                            st.rerun()
                    if (st.session_state.get("active_action") or {}).get("type") == "enhance":
                        st.markdown("**Thinking...**")
                        st.spinner("Thinking...")
                with col_b:
                    if st.button(
                        "Open ticket form",
                        key=ticket_key,
                        use_container_width=True,
                        disabled=st.session_state["is_thinking"],
                    ):
                        st.session_state["open_ticket_ui"] = True

            enhanced = st.session_state.get(enhanced_key)
            if enhanced and not enhanced.get("not_found"):
                st.markdown("**Enhanced answer:**")
                st.markdown(f"{safe_text(enhanced.get('answer',''))}", unsafe_allow_html=True)
                if enhanced.get("citations"):
                    show_key = f"show_sources_enh_{sel_i}"
                    st.session_state.setdefault(show_key, False)
                    col_s, col_t = st.columns([3, 1])
                    with col_s:
                        st.markdown('<div class="stacked-control sources-btn">', unsafe_allow_html=True)
                        if st.button(
                            "Sources (click to expand the list)",
                            key=f"sources_btn_enh_{sel_i}",
                            use_container_width=True,
                            disabled=st.session_state["is_thinking"],
                        ):
                            st.session_state[show_key] = not st.session_state[show_key]
                        st.markdown("</div>", unsafe_allow_html=True)
                    with col_t:
                        if st.button(
                            "Open ticket form",
                            key=f"ticket_btn_enh_{sel_i}",
                            use_container_width=True,
                            disabled=st.session_state["is_thinking"],
                        ):
                            st.session_state["open_ticket_ui"] = True
                    if st.session_state[show_key]:
                        for c in enhanced["citations"]:
                            st.markdown(f"- {safe_text(c)}", unsafe_allow_html=True)
        st.divider()
st.markdown("### Sample questions")
sq1, sq2, sq3 = st.columns(3)
with sq1:
    st.markdown("**AIMA**")
    for i, q in enumerate(AIMA_QUESTIONS, 1):
        st.button(q, on_click=sample_click, args=(q,), key=f"sq_aima_{i}", use_container_width=True, disabled=st.session_state["is_thinking"])
with sq2:
    st.markdown("**Agentic Design Patterns**")
    for i, q in enumerate(AGENTIC_QUESTIONS, 1):
        st.button(q, on_click=sample_click, args=(q,), key=f"sq_agentic_{i}", use_container_width=True, disabled=st.session_state["is_thinking"])
with sq3:
    st.markdown("**Generative AI Design Patterns**")
    for i, q in enumerate(GENAI_QUESTIONS, 1):
        st.button(q, on_click=sample_click, args=(q,), key=f"sq_genai_{i}", use_container_width=True, disabled=st.session_state["is_thinking"])

st.markdown("### Article questions")
st.session_state.setdefault("article_questions", ARTICLE_QUESTIONS_DEFAULT)
aq1, aq2 = st.columns([2, 1])
with aq1:
    for i, q in enumerate(st.session_state["article_questions"], 1):
        label = f"MCP • {q}" if i == 1 else q
        st.button(label, on_click=sample_click, args=(q,), key=f"sq_article_{i}", use_container_width=True, disabled=st.session_state["is_thinking"])
with aq2:
    regen_btn = st.button(
        "Regenerate article questions",
        use_container_width=True,
        disabled=st.session_state["is_thinking"],
    )
    if regen_btn:
        start_action("regen", {})
        st.stop()
        st.rerun()

# Execute active actions after UI is rendered so headers remain visible.
if st.session_state.get("active_action"):
    action = st.session_state["active_action"]
    st.session_state["active_action"] = None
    action_type = action.get("type")
    payload = action.get("payload") or {}
    if action_type == "answer":
        q_norm = (payload.get("question") or "").strip()
        if q_norm:
            push_message("user", q_norm)
            if is_company_question(q_norm):
                answer = company_answer()
                citations = []
                ok = True
            else:
                answer, citations, ok = answer_question(
                    q_norm,
                    book_k=BOOK_K,
                    article_k=ARTICLE_K,
                    enhanced=False,
                )
            if ok:
                push_message("assistant", answer, citations=citations, not_found=False)
            else:
                push_message("assistant", answer, citations=[], not_found=True)
                st.session_state["ticket_prefill"] = {"question": q_norm, "citations": citations}
            st.session_state["last_question"] = q_norm
            st.session_state["last_citations"] = citations
            st.session_state["last_answer"] = answer
    elif action_type == "enhance":
        run_enhance(payload.get("question") or "", payload.get("enhanced_key") or "")
    elif action_type == "regen":
        run_regen()
    st.session_state["is_thinking"] = False
    st.rerun()

def ticket_form(prefill: Optional[Dict]):
    q_pref = ((prefill or {}).get("question") or "").strip()
    c_pref = (prefill or {}).get("citations") or []
    st.write("This will create a GitHub issue in the project repository.")
    user_name = st.text_input("Your name", value="")
    user_email = st.text_input("Your email", value="")
    title_default = "Dataset missing information" if q_pref else "Support request"
    title = st.text_input("Summary (title)", value=title_default)
    desc_default = f"Question:\n{q_pref}\n\nWhat I expected:\n\nWhat happened:\n"
    details = st.text_area("Description (details)", value=(desc_default if q_pref else ""))
    col1, col2 = st.columns(2)
    submit = col1.button("Submit ticket", use_container_width=True)
    cancel = col2.button("Cancel", use_container_width=True)
    if cancel:
        st.session_state["open_ticket_ui"] = False
        return
    if submit:
        if not user_name.strip():
            st.error("Name is required.")
            return
        if not valid_email(user_email):
            st.error("A valid email is required.")
            return
        if not title.strip():
            st.error("Summary (title) is required.")
            return
        if not details.strip():
            st.error("Description is required.")
            return
        body = build_issue_body(user_name.strip(), user_email.strip(), title.strip(), details.strip(), q_pref, c_pref)
        num, err = github_create_issue(title.strip(), body, labels=["support"])
        if err:
            st.error(f"Failed to create GitHub issue: {err}")
        else:
            st.success(f"Ticket created: Issue #{num}")
            st.session_state["open_ticket_ui"] = False

if st.session_state.get("open_ticket_ui"):
    prefill = st.session_state.get("ticket_prefill") or {}
    if hasattr(st, "dialog"):
        @st.dialog("Create support ticket")
        def _dlg():
            ticket_form(prefill)
        _dlg()
    else:
        with st.sidebar.expander("Create support ticket", expanded=True):
            ticket_form(prefill)
