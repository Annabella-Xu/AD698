"""
AD 698 — Section-Scoped Financial RAG
=====================================

Modular pipeline for retrieving and answering questions over SEC 10-K filings
(Items 6, 7, 7A, 8 only). Factored out of the notebook so the same logic is
callable from:

  * the Colab notebook (``AD698_Financial_RAG.ipynb``)
  * the CLI (``python -m src.rag {build,ask,eval}``) — used for Claude Code
  * the Streamlit app (``app.py``)
  * unit tests (not yet added)

Design invariants (do not relax without updating the design brief):

  1. **Hard section scoping.** Retrieval uses cosine similarity over an in-memory
     embedding matrix, then filters candidates against chunk metadata before
     answer generation. The LLM never sees out-of-scope text. This is enforced
     by filtering, not by prompt instruction, because prompt instructions are
     not reliably honored by LLMs.

  2. **Refuse below MIN_SIM.** If the top-1 cosine similarity is below the
     threshold, the system returns ``refused=True`` without calling the LLM.
     This prevents the model from falling back on parametric knowledge.

  3. **Every claim must cite.** The generation prompt requires a strict JSON
     response with ``citations`` listing the ``chunk_id`` backing each claim.
     Answers with no citations are a grounding failure and surface in §5.3.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import html
import json
import os
import random
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Heavy deps (BeautifulSoup, sentence-transformers) are imported lazily inside
# the functions that need them, so `--help` and `ask` stay snappy even before
# the index is built.


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All tunable knobs in one place. Override any field via ``--flag`` on the CLI."""

    # Domain scope — the ONLY SEC Items our RAG system retrieves over.
    # See docs/SCOPE_BOUNDARIES.md for rationale.
    allowed_items: list[str] = field(
        default_factory=lambda: ["Item 6", "Item 7", "Item 7A", "Item 8"]
    )

    # Chunking (token-aware, not character-aware — downstream models bill in tokens)
    chunk_tokens: int = 500
    chunk_overlap: int = 50

    # Embeddings — BGE-small chosen for (a) strong finance performance on MTEB,
    # (b) 384-dim keeps the index small, (c) fully open weights, no API key.
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # Retrieval
    top_k: int = 5
    min_sim: float = 0.20  # below this, refuse rather than hallucinate

    # LLM backend — 'mock' runs deterministically with no API key (useful for CI)
    llm_backend: str = "mock"  # 'openai' | 'gemini' | 'mock'
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-1.5-flash"

    # Paths (filled in at runtime)
    filings_dir: str = "data/filings"
    cache_dir: str = ".cache"

    # Smoke-test knob — process only the first N filings (None = all)
    max_files: Optional[int] = None


CONFIG = Config()


# ---------------------------------------------------------------------------
# M1 — HTML → clean, section-scoped text
# ---------------------------------------------------------------------------

# Every SEC 10-K heading we recognize, in order. We need the full sequence
# even though we only keep 6/7/7A/8 because the NEXT heading is what terminates
# our current section.
ITEM_SEQUENCE = [
    "Item 1", "Item 1A", "Item 1B", "Item 1C", "Item 2", "Item 3", "Item 4",
    "Item 5", "Item 6", "Item 7", "Item 7A", "Item 8", "Item 9", "Item 9A",
    "Item 9B", "Item 9C", "Item 10", "Item 11", "Item 12", "Item 13",
    "Item 14", "Item 15", "Item 16",
]

TABLE_PLACEHOLDER = " [TABLE] "

# Match 'Item 7.', 'ITEM 7 ', 'Item 7 —' — case-insensitive, line-anchored.
def _item_regex(item: str) -> re.Pattern:
    num = item.split()[1]
    return re.compile(rf"(?im)^\s*item\s+{re.escape(num)}\b\s*[\.\-—:]?\s*")


# Header patterns for SEC conforming HTML. We fall back to filename parsing when
# the header is absent (e.g., DIY-exported HTML).
_HEADER_PATTERNS = {
    "cik": re.compile(r"CENTRAL\s*INDEX\s*KEY[^\d]*(\d{4,10})", re.I),
    "company": re.compile(r"COMPANY\s*CONFORMED\s*NAME\s*[:=]?\s*([A-Z0-9 ,.\-&/]+)", re.I),
    "ticker": re.compile(r"TRADING\s*SYMBOL\s*[:=]?\s*([A-Z]{1,6})", re.I),
    "period": re.compile(r"CONFORMED\s*PERIOD\s*OF\s*REPORT\s*[:=]?\s*(\d{8})", re.I),
}


def extract_header(raw_text: str, filename: str) -> dict:
    """Parse (cik, company, ticker, filing_year) from filing header; fall back to filename."""
    hdr = {"cik": None, "company": None, "ticker": None, "filing_year": None}
    # Header is always in the first ~50KB; scanning further wastes time on large filings.
    head = raw_text[:50_000]
    for key, pat in _HEADER_PATTERNS.items():
        m = pat.search(head)
        if not m:
            continue
        val = m.group(1).strip()
        if key == "period":
            hdr["filing_year"] = val[:4]
        else:
            hdr[key] = val

    # Filename fallback — works for both AAPL_10-K_2024.htm and cik0000320193-20240928.htm
    base = os.path.splitext(os.path.basename(filename))[0]
    if not hdr["ticker"]:
        m = re.match(r"^([A-Z]{1,6})[_\-.]", base)
        if m:
            hdr["ticker"] = m.group(1)
    if not hdr["filing_year"]:
        m = re.search(r"(20\d{2})", base)
        if m:
            hdr["filing_year"] = m.group(1)
    if not hdr["cik"]:
        m = re.search(r"cik[-_]?0*(\d{4,10})", base, re.I)
        if m:
            hdr["cik"] = m.group(1)
    return hdr


def html_to_text(raw_html: str) -> str:
    """Convert filing HTML to normalized plain text.

    Tables are replaced with ``[TABLE]`` placeholders because numeric-heavy
    tables flood embedding space with low-semantic tokens and reduce retrieval
    quality on narrative questions. A future iteration should index tables
    separately with a structured-table retriever (TAPAS, Table-Transformer).
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_html, "lxml")
    for t in soup(["script", "style", "head", "meta", "link"]):
        t.decompose()
    for tbl in soup.find_all("table"):
        tbl.replace_with(TABLE_PLACEHOLDER)
    # Preserve block-level breaks so the Item-heading regex can anchor on lines.
    for tag in soup.find_all(["p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
        tag.append("\n")
    text = soup.get_text("\n", strip=False)

    # Unicode normalization (NFKC) folds ligatures and smart-quotes that would
    # otherwise cause embedding mismatch across filings typeset with different fonts.
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_all_item_spans(text: str) -> list[dict]:
    """Every item-heading hit with its character offset and preview."""
    hits = []
    for item in ITEM_SEQUENCE:
        for m in _item_regex(item).finditer(text):
            hits.append({"item": item, "start": m.start(), "preview": text[m.start():m.start() + 200]})
    hits.sort(key=lambda h: h["start"])
    return hits


def _filter_toc_hits(hits: list[dict], text: str, min_body: int = 400) -> list[dict]:
    """Drop table-of-contents entries by requiring real sections to have >= ``min_body`` chars.

    Why 400? Empirically, TOC entries are spaced <100 chars apart; the shortest
    real Item sections (e.g., "Item 6. [RESERVED]") are replaced with a longer
    placeholder pattern and still typically exceed 400 chars. Tune per-corpus
    if you see false drops.
    """
    kept = []
    for i, h in enumerate(hits):
        next_start = hits[i + 1]["start"] if i + 1 < len(hits) else len(text)
        if next_start - h["start"] >= min_body:
            kept.append(h)
    return kept


def extract_items(text: str, wanted: list[str]) -> dict[str, str]:
    """Slice ``text`` into ``{item_name: body}`` for each requested Item.

    Takes the *first real* (post-TOC-filter) occurrence of each Item heading
    and extends until the next real heading.
    """
    hits = _filter_toc_hits(_find_all_item_spans(text), text)
    sections: dict[str, str] = {}
    for i, h in enumerate(hits):
        if h["item"] not in wanted or h["item"] in sections:
            continue
        end = hits[i + 1]["start"] if i + 1 < len(hits) else len(text)
        sections[h["item"]] = text[h["start"]:end].strip()
    return sections


# Per-section cleaning — the signature block and repeated page headers are the
# two biggest sources of retrieval noise we've observed in 10-K filings.
_SIGNATURE_BLOCK = re.compile(r"(?is)(signatures?\s*\n.+?pursuant\s+to\s+the\s+requirements.+?$)")
_BOILERPLATE_PATTERNS = [
    re.compile(r"(?im)^\s*[A-Z][A-Za-z0-9 ,.&\-]+\|\s*20\d{2}\s*Form\s*10-K\s*\|\s*Page\s*\d+\s*$"),
    re.compile(r"(?m)^\s*\d{1,3}\s*$"),  # bare page numbers
    re.compile(r"(?im)^\s*table\s+of\s+contents\s*$"),
]


def clean_section(txt: str) -> str:
    txt = _SIGNATURE_BLOCK.sub("", txt)
    for pat in _BOILERPLATE_PATTERNS:
        txt = pat.sub("", txt)
    txt = re.sub(r"(\[TABLE\]\s*){2,}", "[TABLE] ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return "\n".join(line.rstrip() for line in txt.splitlines()).strip()


# ---------------------------------------------------------------------------
# M2 — Token-aware chunking + metadata
# ---------------------------------------------------------------------------

# Sentence splitter — look-behind for end-punct, look-ahead for capital.
# Good enough for 10-K prose; doesn't need a full NLP library.
_SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z\(\[])")


def _split_sentences(text: str) -> list[str]:
    sents: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        sents.extend(p.strip() for p in _SENT_SPLIT.split(para) if p.strip())
    return sents


def chunk_text(text: str, cfg: Config = CONFIG) -> list[str]:
    """Split text into ~``chunk_tokens`` chunks with ``chunk_overlap`` overlap.

    Algorithm:
      1. Split into sentences.
      2. Pack sentences greedily into chunks capped at ``chunk_tokens``.
      3. Carry the tail of each chunk (up to ``chunk_overlap`` tokens)
         into the next chunk to preserve context across boundaries.
      4. Ultra-long sentences (>cap) fall back to word windowing — rare in
         10-Ks but happens with malformed sentence-end punctuation.

    Token counting uses whitespace word count (~0.75 GPT-tokens per word).
    Good enough for chunking decisions without the tiktoken dependency.
    """
    tok_len = lambda s: len(s.split())

    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0

    for sent in _split_sentences(text):
        st = tok_len(sent)

        if st > cfg.chunk_tokens:
            # Emit current buffer then window the long sentence.
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_tok = [], 0
            words = sent.split()
            step = max(1, cfg.chunk_tokens - cfg.chunk_overlap)
            for i in range(0, len(words), step):
                chunks.append(" ".join(words[i:i + cfg.chunk_tokens]))
            continue

        if cur_tok + st > cfg.chunk_tokens and cur:
            chunks.append(" ".join(cur))
            # Tail carry — walk backwards collecting sentences until we've
            # accumulated ~overlap tokens of context for the next chunk.
            tail: list[str] = []
            tail_tok = 0
            for t in reversed(cur):
                tlen = tok_len(t)
                if tail_tok + tlen > cfg.chunk_overlap:
                    break
                tail.insert(0, t)
                tail_tok += tlen
            cur, cur_tok = tail, tail_tok

        cur.append(sent)
        cur_tok += st

    if cur:
        chunks.append(" ".join(cur))
    return chunks


def build_chunks(section_rows: list[dict], cfg: Config = CONFIG) -> list[dict]:
    """Attach full provenance metadata to every chunk."""
    out: list[dict] = []
    for r in section_rows:
        for i, piece in enumerate(chunk_text(r["text"], cfg)):
            # Deterministic chunk_id — stable across runs IF the upstream
            # chunking is deterministic. Re-chunking invalidates labels.
            h = hashlib.md5(f"{r.get('cik')}|{r['item']}|{i}|{piece[:40]}".encode()).hexdigest()[:10]
            cid = f"{r.get('ticker') or r.get('cik') or 'UNK'}-{r['item'].replace(' ', '')}-{i:03d}-{h}"
            out.append({
                "chunk_id": cid,
                "cik": r.get("cik"),
                "company": r.get("company"),
                "ticker": r.get("ticker"),
                "filing_year": r.get("filing_year"),
                "item": r["item"],
                "chunk_index": i,
                "token_count": len(piece.split()),
                "text": piece,
            })
    return out


# ---------------------------------------------------------------------------
# M3 — Embedding + cosine retrieval
# Uses an in-memory numpy matrix and sklearn cosine_similarity. No FAISS —
# our corpus (tens of thousands of chunks) is well within numpy's reach.
# ---------------------------------------------------------------------------

def _embed_corpus(chunks: list[dict], cfg: Config) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.embed_model)
    emb = model.encode(
        [c["text"] for c in chunks],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine-ready: inner product == cosine similarity
    )
    return np.asarray(emb, dtype="float32")


@dataclass
class Index:
    """A loaded embedding matrix + metadata, ready to query."""
    matrix: np.ndarray  # shape (n_chunks, dim), L2-normalized
    chunks_by_id: dict[str, dict]
    chunks_meta: list[dict]
    cfg: Config
    embed_model_obj: object = None  # SentenceTransformer — lazy-loaded in retrieve()


def build_index(filings_dir: str, cache_dir: str, cfg: Config = CONFIG) -> Index:
    """Run M1→M3 end-to-end: parse filings, chunk, embed, build cosine index.

    Caches all intermediate artifacts under ``cache_dir`` so subsequent runs
    skip the expensive embedding step.
    """
    os.makedirs(cache_dir, exist_ok=True)
    sections_path = os.path.join(cache_dir, "sections.jsonl")
    chunks_path = os.path.join(cache_dir, "chunks.jsonl")
    emb_path = os.path.join(cache_dir, "embeddings.npy")
    meta_path = os.path.join(cache_dir, "chunk_meta.jsonl")

    # --- M1: extraction ---
    if os.path.exists(sections_path):
        with open(sections_path) as f:
            section_rows = [json.loads(l) for l in f]
        print(f"[cache] loaded {len(section_rows)} sections")
    else:
        paths: list[str] = []
        for pat in ("*.htm", "*.html", "*.HTM", "*.HTML"):
            paths.extend(glob.glob(os.path.join(filings_dir, "**", pat), recursive=True))
        paths = sorted(set(paths))
        if not paths:
            raise FileNotFoundError(f"No HTML filings found under {filings_dir}")
        if cfg.max_files is not None:
            paths = paths[: cfg.max_files]
            print(f"[smoke-test] max_files={cfg.max_files} — processing only first {len(paths)} filings")
        print(f"[build] extracting Items {cfg.allowed_items} from {len(paths)} filings…")
        section_rows = []
        for p in paths:
            with open(p, "rb") as f:
                raw = f.read().decode("utf-8", errors="replace")
            hdr = extract_header(raw, p)
            text = html_to_text(raw)
            for item, body in extract_items(text, cfg.allowed_items).items():
                clean = clean_section(body)
                if len(clean) < 500:
                    continue  # skip near-empty Items (e.g. "Not applicable.")
                section_rows.append({
                    **hdr, "file": os.path.basename(p), "item": item,
                    "char_count": len(clean), "text": clean,
                })
        with open(sections_path, "w") as f:
            for r in section_rows:
                f.write(json.dumps(r) + "\n")
        print(f"[build] wrote {len(section_rows)} sections")

    # --- M2: chunking ---
    if os.path.exists(chunks_path):
        with open(chunks_path) as f:
            chunks = [json.loads(l) for l in f]
        print(f"[cache] loaded {len(chunks):,} chunks")
    else:
        chunks = build_chunks(section_rows, cfg)
        with open(chunks_path, "w") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        print(f"[build] wrote {len(chunks):,} chunks")

    # --- M3: embeddings ---
    def _load_validated_embeddings() -> np.ndarray | None:
        """Use cached embeddings only when they match the current chunk IDs."""
        if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
            return None
        emb = np.load(emb_path)
        with open(meta_path) as f:
            cached_meta = [json.loads(l) for l in f]
        current_ids = [c["chunk_id"] for c in chunks]
        cached_ids = [m.get("chunk_id") for m in cached_meta]
        if emb.shape[0] != len(current_ids):
            print("[cache] ignored embeddings: row count does not match current chunks")
            return None
        if current_ids != cached_ids:
            print("[cache] ignored embeddings: chunk IDs changed")
            return None
        return emb

    embeddings = _load_validated_embeddings()
    if embeddings is not None:
        print(f"[cache] loaded validated embeddings with {embeddings.shape[0]:,} vectors")
    else:
        embeddings = _embed_corpus(chunks, cfg)
        np.save(emb_path, embeddings)
        # Persist metadata without text so cache validation can detect stale chunks
        with open(meta_path, "w") as f:
            for c in chunks:
                f.write(json.dumps({k: v for k, v in c.items() if k != "text"}) + "\n")
        print(f"[build] built embedding matrix with {embeddings.shape[0]:,} vectors")

    return Index(
        matrix=embeddings,
        chunks_by_id={c["chunk_id"]: c for c in chunks},
        chunks_meta=chunks,  # full records; memory-cheap for this corpus size
        cfg=cfg,
    )


def retrieve(index: Index, query: str, allowed_items: list[str] | None = None,
             k: int | None = None, company: str | None = None,
             ticker: str | None = None, cik: str | None = None) -> list[dict]:
    """Return top-k chunks filtered by SEC Item and optional firm metadata.

    This prevents cross-firm contamination: for firm-specific questions, pass
    company, ticker, or CIK so ranking happens only inside that firm's chunks.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    cfg = index.cfg
    allowed = set(allowed_items or cfg.allowed_items)
    k = k or cfg.top_k
    if index.embed_model_obj is None:
        index.embed_model_obj = SentenceTransformer(cfg.embed_model)

    q_vec = index.embed_model_obj.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q_vec, index.matrix)[0]

    candidate_ids: list[int] = []
    for i, meta in enumerate(index.chunks_meta):
        if meta.get("item") not in allowed:
            continue
        if company is not None and meta.get("company") != company:
            continue
        if ticker is not None and meta.get("ticker") != ticker:
            continue
        if cik is not None and str(meta.get("cik")) != str(cik):
            continue
        candidate_ids.append(i)

    if not candidate_ids:
        return []

    candidate_ids_arr = np.array(candidate_ids)
    ranked = candidate_ids_arr[np.argsort(-sims[candidate_ids_arr])[:k]]

    results: list[dict] = []
    for vec_id in ranked:
        meta = index.chunks_meta[int(vec_id)]
        results.append({
            "score": float(sims[int(vec_id)]),
            "chunk_id": meta["chunk_id"],
            "item": meta["item"],
            "company": meta.get("company"),
            "ticker": meta.get("ticker"),
            "cik": meta.get("cik"),
            "year": meta.get("filing_year"),
            "text": meta["text"],
        })
    return results


# ---------------------------------------------------------------------------
# M4 — Grounded generation
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are a careful financial analyst RAG system restricted to SEC 10-K 2024 filings.
Your job: answer the user's question USING ONLY the CONTEXT blocks below. Do NOT use outside knowledge.

Rules:
- If the context does not contain enough evidence, set "refused": true and explain what is missing.
- Every factual statement in your answer must be backed by at least one chunk_id in "citations".
- Cite the Item number (e.g. "Item 7") and the chunk_id next to each claim.
- Output MUST be strict JSON: {"answer": "...", "citations": [{"chunk_id": "...", "item": "..."}], "refused": false}.
- Do not invent chunk_ids. Only cite chunks that appear in the CONTEXT.
- Retrieved sections are limited to: Item 6, Item 7, Item 7A, Item 8. Refuse any question that requires other items.
"""


def _build_prompt(question: str, hits: list[dict]) -> tuple[str, str]:
    blocks = [
        f"[chunk_id={h['chunk_id']}] [item={h['item']}] "
        f"[company={h['company']}] [year={h['year']}]\n{h['text']}"
        for h in hits
    ]
    user = f"CONTEXT:\n{chr(10).join(f'---{chr(10)}{b}' for b in blocks)}\n\nQUESTION: {question}\n\nRespond as strict JSON only."
    return RAG_SYSTEM_PROMPT, user


def _parse_json_answer(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception as e:
        # Failed parse is a grounding failure — surface as refused, not crash.
        return {"answer": raw, "citations": [], "refused": True, "error": f"JSON parse failed: {e}"}


def _llm_generate(system: str, user: str, cfg: Config) -> dict:
    if cfg.llm_backend == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=cfg.openai_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return _parse_json_answer(resp.choices[0].message.content)

    if cfg.llm_backend == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(cfg.gemini_model, system_instruction=system)
        resp = model.generate_content(
            user,
            generation_config={"temperature": 0, "response_mime_type": "application/json"},
        )
        return _parse_json_answer(resp.text)

    # Mock backend — deterministic, no API key, cites first 2 hits.
    ids = re.findall(r"chunk_id=([A-Za-z0-9\-]+)", user)
    items = re.findall(r"item=(Item \S+)", user)
    return {
        "answer": "[MOCK] No LLM backend configured. Set LLM_BACKEND=openai|gemini for real answers.",
        "citations": [{"chunk_id": c, "item": it} for c, it in list(zip(ids, items))[:2]],
        "refused": False,
    }




def _validate_citations(out: dict, hits: list[dict]) -> dict:
    """Only keep citations that point to retrieved chunks.

    If the model invents a chunk_id or returns no valid citation for a
    non-refused answer, force refusal so citation coverage is honest.
    """
    valid = {h["chunk_id"]: h for h in hits}
    cleaned = []
    for c in out.get("citations", []) or []:
        cid = c.get("chunk_id")
        if cid in valid:
            h = valid[cid]
            cleaned.append({
                "chunk_id": cid,
                "item": h["item"],
                "company": h.get("company"),
                "ticker": h.get("ticker"),
                "score": h.get("score"),
            })
    out["citations"] = cleaned
    if not out.get("refused") and not cleaned:
        out["refused"] = True
        out["answer"] = (
            "The model did not provide valid citations from the retrieved context, "
            "so the answer is refused."
        )
    return out

def rag_answer(index: Index, question: str, allowed_items: list[str] | None = None,
               k: int | None = None, company: str | None = None,
               ticker: str | None = None, cik: str | None = None) -> dict:
    """Full pipeline: retrieve → refuse-or-generate → return cited JSON."""
    hits = retrieve(
        index,
        question,
        allowed_items=allowed_items,
        k=k,
        company=company,
        ticker=ticker,
        cik=cik,
    )
    if not hits or hits[0]["score"] < index.cfg.min_sim:
        return {
            "question": question,
            "answer": "Retrieved evidence below similarity threshold — unable to answer.",
            "citations": [],
            "refused": True,
            "retrieved": hits,
        }
    system, user = _build_prompt(question, hits)
    out = _llm_generate(system, user, index.cfg)
    out.setdefault("citations", [])
    out.setdefault("refused", False)
    out = _validate_citations(out, hits)
    out["question"] = question
    out["retrieved"] = hits
    return out


# ---------------------------------------------------------------------------
# M5 — Evaluation
# ---------------------------------------------------------------------------

def evaluate(index: Index, labeled_pairs_path: str, k: int | None = None) -> dict:
    """Compute Hit@k, citation coverage, and leakage on the labeled pair set.

    ``labeled_pairs_path`` is a CSV with columns:
      qid, question, gold_item, gold_company, gold_chunk_id_contains

    Hit@k is binary per question (1 if any retrieved chunk_id contains the
    gold substring, else 0). Report the mean across questions.
    """
    k = k or index.cfg.top_k
    df = pd.read_csv(labeled_pairs_path)
    required = {"qid", "question", "gold_item", "gold_chunk_id_contains"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labeled_pairs missing columns: {missing}")

    rows: list[dict] = []
    for _, r in df.iterrows():
        if not isinstance(r["gold_chunk_id_contains"], str) or not r["gold_chunk_id_contains"]:
            continue  # un-labeled rows — skip rather than scoring as miss
        firm = r.get("gold_company") if "gold_company" in r and isinstance(r.get("gold_company"), str) and r.get("gold_company").strip() else None
        hits = retrieve(index, r["question"], allowed_items=[r["gold_item"]], k=k, company=firm)
        gold_id = r["gold_chunk_id_contains"]
        hit = any(h["chunk_id"] == gold_id or gold_id in h["chunk_id"] for h in hits)
        rows.append({
            "qid": r["qid"],
            "gold_item": r["gold_item"],
            "hit_at_k": int(hit),
            "top1_chunk": hits[0]["chunk_id"] if hits else None,
            "top1_score": round(hits[0]["score"], 3) if hits else None,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return {"hit_at_k": None, "n_labeled": 0, "per_question": []}

    return {
        "hit_at_k": float(out["hit_at_k"].mean()),
        "n_labeled": len(out),
        "per_item": out.groupby("gold_item")["hit_at_k"].mean().round(3).to_dict(),
        "per_question": out.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _cmd_build(args) -> int:
    cfg = _cfg_from_args(args)
    build_index(args.filings_dir, args.cache_dir, cfg)
    print("OK")
    return 0


def _cmd_ask(args) -> int:
    cfg = _cfg_from_args(args)
    index = build_index(args.filings_dir, args.cache_dir, cfg)
    out = rag_answer(index, args.question)
    # Drop retrieved-hits payload for clean terminal output
    printable = {k: v for k, v in out.items() if k != "retrieved"}
    printable["retrieved_chunk_ids"] = [h["chunk_id"] for h in out.get("retrieved", [])]
    print(json.dumps(printable, indent=2))
    return 0


def _cmd_eval(args) -> int:
    cfg = _cfg_from_args(args)
    index = build_index(args.filings_dir, args.cache_dir, cfg)
    result = evaluate(index, args.labeled_pairs, k=cfg.top_k)
    print(json.dumps({k: v for k, v in result.items() if k != "per_question"}, indent=2))
    if args.verbose and result.get("per_question"):
        print("\nPer-question:")
        for r in result["per_question"]:
            marker = "✓" if r["hit_at_k"] else "✗"
            print(f"  {marker} {r['qid']} [{r['gold_item']}] top1={r['top1_score']} chunk={r['top1_chunk']}")
    return 0


def _cfg_from_args(args) -> Config:
    cfg = Config()
    for k in ("chunk_tokens", "chunk_overlap", "top_k", "min_sim", "embed_model", "llm_backend", "max_files"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(cfg, k, v)
    cfg.filings_dir = args.filings_dir
    cfg.cache_dir = args.cache_dir
    return cfg


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--filings-dir", default="data/filings", help="directory of 10-K HTML files")
    p.add_argument("--cache-dir", default=".cache", help="where intermediate artifacts live")
    p.add_argument("--chunk-tokens", type=int, help="override chunk size")
    p.add_argument("--chunk-overlap", type=int, help="override chunk overlap")
    p.add_argument("--top-k", type=int, help="retrieval k")
    p.add_argument("--min-sim", type=float, help="refuse-below-this cosine threshold")
    p.add_argument("--embed-model", help="sentence-transformers model name")
    p.add_argument("--llm-backend", choices=["openai", "gemini", "mock"], help="LLM for generation")
    p.add_argument("--max-files", type=int, help="smoke-test: process only first N filings")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.rag",
        description="Section-scoped RAG over SEC 10-K filings (Items 6/7/7A/8).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="parse → chunk → embed → index (cached)")
    _add_common(p_build)
    p_build.set_defaults(fn=_cmd_build)

    p_ask = sub.add_parser("ask", help="ask a single question")
    _add_common(p_ask)
    p_ask.add_argument("question", help="the question to answer")
    p_ask.set_defaults(fn=_cmd_ask)

    p_eval = sub.add_parser("eval", help="Hit@k on the labeled pair set")
    _add_common(p_eval)
    p_eval.add_argument("--labeled-pairs", default="data/labeled_pairs.csv")
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.set_defaults(fn=_cmd_eval)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
