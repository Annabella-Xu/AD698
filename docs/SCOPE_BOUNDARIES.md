# Scope Boundaries

> **Paste this into the design brief as a highlighted box in §3 (before §3.1).**
> This is the single most important clarification for graders and end-users
> so they do not expect capabilities the system was not designed to provide.

---

## ✅ In scope — this system answers

| Category | Example questions | Source Items |
|---|---|---|
| Narrative financial performance | "What drove revenue growth YoY?" | Item 7 (MD&A) |
| Liquidity & capital resources | "How much runway does the firm have?" | Item 7 |
| Quantified market risk disclosures | "What is the ±100bp interest-rate sensitivity?" | Item 7A |
| Accounting-policy narrative | "What revenue recognition policy applies?" | Item 8 footnotes |
| Critical accounting estimates | "What impairment tests were performed?" | Items 7 & 8 |

## ❌ Out of scope — by design

| Category | Why excluded | What graders should know |
|---|---|---|
| **Precise numeric lookups** ("What was Q4 EPS?") | Item 8 tables are replaced with `[TABLE]` placeholders during cleaning; numeric-heavy tables flood embeddings with low-semantic tokens and degrade retrieval on narrative questions | A follow-up iteration should index tables separately with a structured-table retriever (TAPAS, Table-Transformer) |
| **Cross-firm comparisons** ("Which firm grew fastest?") | Each answer is scoped to one firm; the system does not perform aggregation across filings | Users must query firm-by-firm; aggregation is an analytics-layer concern, not RAG |
| **Governance, compensation, cyber** (Items 1/1A/10/11) | Domain track is *Financial Performance & Risk*; these items fall outside the allow-list | Enforced by hard filter in `retrieve()` — LLM never sees out-of-scope text, even when the query asks for it |
| **Forward-looking guidance beyond what's in the filing** | LLM parametric knowledge is suppressed by refusal-below-MIN_SIM and strict citation contract | System returns `refused=true` if retrieved evidence does not cover the question |

## 🔒 Enforcement mechanism

Section-scoping is implemented as a **post-retrieval hard filter** on chunk
metadata, not as a prompt instruction:

```python
# From src/rag.py:retrieve
for score, vec_id in zip(sims[0], ids[0]):
    meta = index.chunks_meta[vec_id]
    if meta["item"] not in allowed:
        continue   # HARD section-scoping — never relaxed
    ...
```

Prompt-only scoping is not reliable because LLMs do not always honor
"only use these sources" instructions. Filtering at the retrieval layer
means the model physically cannot see out-of-scope chunks.
