# AD 698 — Section-Scoped RAG over SEC 10-K (2024)

**Course**: *Applied Generative AI for Business Analytics* — Boston University
**Domain Track**: Financial Performance & Risk Analysis
**SEC Items in scope**: 6 (Selected Financial Data) · 7 (MD&A) · 7A (Market Risk) · 8 (Financial Statements)

A retrieval-augmented generation system that answers analyst questions over FY-2024 SEC 10-K filings,
with **hard section-scoping** (retrieval is post-filtered to allowed Items — the LLM never sees
out-of-scope text) and a **strict JSON citation contract** (every answer traces to a `chunk_id`).

---

## Quick start

### Option A — Colab (Milestones 1–5 end-to-end)

1. Put your SEC 10-K HTML files in `MyDrive/SEC-10K-2024-HTML/`.
2. Open [`AD698_Financial_RAG.ipynb`](AD698_Financial_RAG.ipynb) in Colab → Runtime → Run all.
3. Set `LLM_BACKEND` in cell *0.4 Configuration*:
   - `'mock'` — deterministic stub, no API key (dry run)
   - `'openai'` — needs `OPENAI_API_KEY` in Colab Secrets
   - `'gemini'` — needs `GEMINI_API_KEY` in Colab Secrets
4. Review outputs under `AD698_outputs/`.

### Option B — Claude Code / local CLI

```bash
# one-time setup
pip install -r requirements.txt

# put your filings here (override with --filings-dir)
mkdir -p data/filings
cp your-10k-files/*.htm data/filings/

# run the full pipeline: extract → chunk → embed → cosine retrieval index
python -m src.rag build --filings-dir data/filings --cache-dir .cache

# interactive Q&A
python -m src.rag ask "What does management say about liquidity headed into 2025?"

# evaluate against the labeled set
python -m src.rag eval --labeled-pairs data/labeled_pairs.csv
```

### Option C — Streamlit deployment (bonus)

```bash
streamlit run app.py
```
Open the browser, pick a firm, ask a question, and see citations + the thumbs up/down feedback
logger (writes to `.cache/feedback.csv`).

---

## Repository layout

```
.
├── AD698_Financial_RAG.ipynb    # end-to-end notebook (Milestones 1–5)
├── app.py                        # Streamlit UI (25-pt bonus)
├── src/
│   ├── __init__.py
│   └── rag.py                    # modular pipeline: CLI + library
├── data/
│   ├── domain_questions.csv      # 15 structured domain questions
│   └── labeled_pairs.csv         # human-labeling worksheet for Hit@k
├── docs/
│   ├── SCOPE_BOUNDARIES.md       # scope box — paste into design brief §3
│   ├── EVALUATION_PLAN.md        # expanded §9 with targets + baselines
│   ├── ABLATION_TEMPLATE.md      # chunk / similarity sweep template
│   └── LABELING_GUIDE.md         # how to human-label the eval pairs
├── requirements.txt
├── .gitignore
└── README.md                     # this file
```

---

## Configuration knobs (`src/rag.py` and notebook cell *0.4*)

| Variable | Default | What it does |
|---|---|---|
| `ALLOWED_ITEMS` | `['Item 6', 'Item 7', 'Item 7A', 'Item 8']` | Hard allow-list applied after cosine search and before generation |
| `CHUNK_TOKENS` | `500` | Approximate token target per chunk, using whitespace-token estimates |
| `CHUNK_OVERLAP` | `50` | Approximate token overlap between consecutive chunks |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | 384-dim open embedding model |
| `TOP_K` | `5` | Chunks returned per query |
| `MIN_SIM` | `0.20` | Refuse if top-1 cosine < this threshold |
| `LLM_BACKEND` | `'mock'` | `'openai'` / `'gemini'` / `'mock'` |

---

## Milestone map

| Milestone | Deliverable | Location |
|---|---|---|
| **M1** | Item 6/7/7A/8 extraction + cleaning (TOC bleed, signatures, tables → placeholder) | notebook §1 · `src/rag.py:extract_items` |
| **M2** | Approximate token-aware chunking (≈500 tokens, 50 overlap) with full provenance metadata | notebook §2 · `src/rag.py:chunk_text` |
| **M3** | BGE-small embeddings + in-memory embedding matrix + cosine-similarity retrieval with strict Item filtering | notebook §3 · `src/rag.py:retrieve` |
| **M4** | Grounded generation: JSON contract `{answer, citations, refused}`; refusal below `MIN_SIM` | notebook §4 · `src/rag.py:rag_answer` |
| **M5** | Eval: Hit@k vs. human labels, citation coverage, cross-Item leakage, manual hallucination review | notebook §5 · `src/rag.py:evaluate` |

---

## What's new vs. the initial draft

Recent improvements (addressing reviewer feedback):

1. **Prominent scope-boundaries box** (notebook §0.5, [`docs/SCOPE_BOUNDARIES.md`](docs/SCOPE_BOUNDARIES.md)) — numeric table lookups and out-of-scope Items called out explicitly.
2. **Human-labeling worksheet** (`data/labeled_pairs.csv` + [`docs/LABELING_GUIDE.md`](docs/LABELING_GUIDE.md)) — replaces the auto-populated ground truth. Hit@k now reflects a curated reference set, not a circular self-label.
3. **Real cross-domain leakage probe** (notebook §5.3b) — a query that has both in-scope *and* out-of-scope answers; asserts citations only contain allowed Items.
4. **Manual hallucination review workflow** (notebook §5.4) — replaces the same-model LLM-as-judge with a 20-answer manual audit that produces a supported-claim rate with a binomial CI.
5. **Ablation sweep** (notebook §5.5) — `CHUNK_TOKENS × MIN_SIM` grid, justifies the defaults.
6. **Explicit success criteria** ([`docs/EVALUATION_PLAN.md`](docs/EVALUATION_PLAN.md)) — Hit@5 ≥ 60%, citation coverage ≥ 85%, cross-Item leakage = 0, hallucination < 20%.
7. **Modular `src/rag.py`** — the notebook logic is also exposed as a library + CLI so the system is runnable from a terminal (Claude Code) or wrapped in a Streamlit app.
8. **`app.py` Streamlit bonus** — full UI with feedback logging (question, answer, citations, thumbs ± → CSV).

---

## Citation format

Every non-refused answer returns strict JSON:

```json
{
  "answer": "Management attributes revenue movement to disclosed drivers in the cited Item 7 chunks.",
  "citations": [
    {"chunk_id": "AAPL-Item7-012-a3b9c2", "item": "Item 7"},
    {"chunk_id": "AAPL-Item7-013-8f1de4", "item": "Item 7"}
  ],
  "refused": false
}
```

---

## License & credits

Academic project for AD 698 (BU MSBA). SEC filings are public-domain.
Embeddings: [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5).
