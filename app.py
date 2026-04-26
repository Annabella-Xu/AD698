"""
AD 698 Financial RAG — Streamlit UI (25-pt deployment bonus)
============================================================

Launch locally:
    streamlit run app.py

What it does:
  1. Builds (or loads from cache) the section-scoped FAISS index.
  2. Gives the user a question box + optional firm filter.
  3. Displays the JSON answer with clickable citations showing the underlying
     chunk text so graders can verify grounding.
  4. Logs (timestamp, question, answer, citations, thumbs ±) to ``.cache/feedback.csv``
     so we can compute a usage approval rate — the success metric proposed in
     docs/EVALUATION_PLAN.md §3.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.rag import CONFIG, build_index, rag_answer, Config


# ---------------------------------------------------------------------------
# Config — read from env first so deployments can override without code edits.
# ---------------------------------------------------------------------------
FILINGS_DIR = os.environ.get("AD698_FILINGS_DIR", "data/filings")
CACHE_DIR = os.environ.get("AD698_CACHE_DIR", ".cache")
FEEDBACK_LOG = Path(CACHE_DIR) / "feedback.csv"

st.set_page_config(page_title="AD698 Financial RAG", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Index is expensive to build; cache it across reruns.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading index (one-time)…")
def _load_index(filings_dir: str, cache_dir: str, backend: str, min_sim: float):
    cfg = Config()
    cfg.filings_dir = filings_dir
    cfg.cache_dir = cache_dir
    cfg.llm_backend = backend
    cfg.min_sim = min_sim
    return build_index(filings_dir, cache_dir, cfg)


# ---------------------------------------------------------------------------
# Sidebar — config knobs visible so graders can demonstrate the ablation
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ System config")
backend = st.sidebar.selectbox("LLM backend", ["mock", "openai", "gemini"], index=0,
                               help="'mock' requires no API key. Choose openai/gemini for real answers.")
min_sim = st.sidebar.slider("MIN_SIM (refuse below)", 0.0, 0.6, 0.20, 0.05,
                            help="Top-1 cosine below this threshold → refused.")
st.sidebar.caption(f"Filings: `{FILINGS_DIR}`")
st.sidebar.caption(f"Cache:   `{CACHE_DIR}`")

# Defensive check so the UI doesn't crash on a fresh clone with no filings.
if not os.path.isdir(FILINGS_DIR) or not any(Path(FILINGS_DIR).rglob("*.htm*")):
    st.error(
        f"No 10-K HTML filings found under `{FILINGS_DIR}`. "
        "Drop some filings there (or set `AD698_FILINGS_DIR`) and reload."
    )
    st.stop()

index = _load_index(FILINGS_DIR, CACHE_DIR, backend, min_sim)

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("📄 AD 698 — Section-Scoped Financial RAG")
st.caption(
    "Ask an analyst-style question about any firm in the loaded 10-K corpus. "
    "Retrieval is hard-filtered to **Items 6 / 7 / 7A / 8** — no other sections are ever shown to the LLM."
)

companies = sorted({m.get("company") for m in index.chunks_meta if m.get("company")})
col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input(
        "Your question",
        placeholder="What does management say about liquidity into 2025?",
    )
with col2:
    firm = st.selectbox("Filter by firm (optional)", ["(any)"] + companies)

ask = st.button("Ask", type="primary", disabled=not question.strip())

# ---------------------------------------------------------------------------
# Answer + citation panel
# ---------------------------------------------------------------------------
if ask:
    # Firm-scope the question by appending a suffix — the embedding model picks
    # up the context so results favor that firm's chunks. The hard Item filter
    # still applies independently.
    scoped_q = question if firm == "(any)" else f"{question} (Firm: {firm})"
    with st.spinner("Retrieving and generating…"):
        result = rag_answer(index, scoped_q)

    st.subheader("Answer")
    if result.get("refused"):
        st.warning(result.get("answer", "System refused the question."))
    else:
        st.success(result.get("answer", "(no answer)"))

    st.subheader("Citations")
    if not result.get("citations"):
        st.info("No citations returned.")
    else:
        for c in result["citations"]:
            cid = c.get("chunk_id")
            item = c.get("item")
            # Re-join the full chunk text for verification.
            chunk = index.chunks_by_id.get(cid, {})
            with st.expander(f"🔖 {cid}  ·  {item}  ·  {chunk.get('company','')}"):
                st.write(chunk.get("text", "(chunk text not available)"))

    st.subheader("Retrieved chunks (top-k)")
    for h in result.get("retrieved", []):
        with st.expander(f"score={h['score']:.3f}  ·  {h['chunk_id']}  ·  {h['item']}  ·  {h['company']}"):
            st.write(h["text"])

    # Feedback logger — critical for the success metric in docs/EVALUATION_PLAN.md §3.
    st.subheader("Rate this answer")
    colA, colB = st.columns(2)
    vote = None
    if colA.button("👍 helpful"):
        vote = 1
    if colB.button("👎 not helpful"):
        vote = 0
    if vote is not None:
        FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        new_file = not FEEDBACK_LOG.exists()
        with FEEDBACK_LOG.open("a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "question", "firm", "refused", "vote",
                            "answer", "citation_chunk_ids"])
            w.writerow([
                datetime.utcnow().isoformat(),
                scoped_q,
                firm,
                result.get("refused", False),
                vote,
                (result.get("answer") or "")[:500],
                ";".join(c.get("chunk_id", "") for c in result.get("citations", [])),
            ])
        st.toast(f"Logged feedback → {FEEDBACK_LOG}")

# ---------------------------------------------------------------------------
# Corpus stats footer — handy during demos
# ---------------------------------------------------------------------------
with st.sidebar.expander("Corpus stats"):
    st.write(f"Chunks indexed: **{len(index.chunks_meta):,}**")
    st.write(f"Firms: **{len(companies)}**")
    st.write(f"Items in scope: {', '.join(index.cfg.allowed_items)}")
