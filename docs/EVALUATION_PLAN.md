# Evaluation Plan

> **Replaces §9 of the design brief.** Detailed enough to be actionable at M5;
> explicit targets so graders can check pass/fail objectively.

---

## 1. Dimensions measured

We evaluate along four **independent** axes. Each has a single headline metric
and a hard-or-soft success criterion. Axes are independent — a system can pass
retrieval and fail grounding; the report must state each separately.

| # | Axis | Metric | Target |
|---|---|---|---|
| 1 | Retrieval accuracy | **Hit@5** on 25 human-labeled (question, gold_chunk_id) pairs | **≥ 60%** (vs. ~10% random baseline) |
| 2 | Grounding reliability | **Citation coverage** = fraction of non-refused answers with ≥ 1 citation | **≥ 85%** |
| 3 | Section integrity | **Cross-Item leakage** = # citations whose `item ∉ ALLOWED_ITEMS` | **= 0** (hard requirement) |
| 4 | Answer faithfulness | **Supported-claim rate** via manual review of 20 answers | **≥ 80%** supported (i.e., hallucination < 20%) |

---

## 2. Axis 1 — Retrieval accuracy (Hit@5)

**Ground truth**: 25 human-labeled pairs in `data/labeled_pairs.csv`. Labeling
protocol is in [`docs/LABELING_GUIDE.md`](LABELING_GUIDE.md). Each pair specifies
a question, the gold SEC Item, the firm, and a distinctive substring of the
gold chunk's id.

**Metric**: binary hit — for each question, did any of the top-5 retrieved
chunks match the gold chunk_id substring? Mean over all questions is Hit@5.
Because each question has one gold chunk, Hit@5 here equals Recall@5.

**Baseline**: random-Item retrieval (query with a random allowed Item that
is not the gold Item). Typical baseline Hit@5 is 5–15% — our system must
exceed this by **at least 4×**.

**Reporting**: `python -m src.rag eval --verbose` prints per-item Hit@5
so weaknesses (e.g., Item 8 worse than Item 7) surface clearly.

---

## 3. Axis 2 — Grounding reliability (citation coverage)

**Metric**: of all non-refused answers on the 15-question × N-firm grid,
what fraction cite at least one `chunk_id`? Computed in notebook §5.3.

**Target**: ≥ 85%. Rationale: every claim must be traceable. Answers with no
citations indicate either (a) the LLM ignored the JSON contract or (b) the
context was too thin to justify a citation. Both are grounding failures.

**Mitigation if below target**: tighten the system prompt with a
*negative example* ("Do not output an answer with an empty citations array"),
and raise `MIN_SIM` so the retrieval layer refuses earlier.

---

## 4. Axis 3 — Section integrity (leakage = 0)

**Metric**: sweep every generated answer's `citations[]`, assert every
`item` ∈ `ALLOWED_ITEMS`. Count violations.

**Target**: **zero**. This is a hard requirement, not a soft one — any
leak is a design-contract violation.

**Stress probe** (notebook §5.3b): a query with *both* an in-scope and an
out-of-scope natural answer — e.g., *"What risks has the board identified
regarding the company's derivative program?"* (answerable from Item 7A
about the derivative program; also answerable from Item 10 about board
oversight — but Item 10 is out of scope). Assert citations only contain
allowed Items. If this test passes, the hard-filter is working.

---

## 5. Axis 4 — Answer faithfulness (manual hallucination review)

We **do not** use LLM-as-judge because the judge shares a backend with the
generator, making it an unreliable check on its own hallucinations. Instead:

**Protocol** (notebook §5.4):

1. Sample 20 non-refused answers across the 15-question × N-firm grid.
2. For each, display: question, answer, cited chunk text.
3. Labeler marks each factual claim in the answer as **supported** /
   **unsupported** based only on the cited chunks.
4. Compute **supported-claim rate** = supported / total claims.

**Target**: ≥ 80% supported. Report a 95% binomial confidence interval
(exact / Clopper–Pearson) because n = 20 gives a wide CI.

**Reporting**: document unsupported-claim patterns (e.g., *"model extrapolates
trends beyond the disclosed period"*) so the final reflection section points
to concrete future work.

---

## 6. Experimental rigor

- **Seeds fixed**: `random.seed(42)` before sampling for manual review.
- **Temperature 0** for all generation so the eval is reproducible.
- **Cache hygiene**: clear `.cache/` before the final eval run so upstream
  bugs (e.g., stale chunk ids) cannot silently affect results.
- **Ablations**: see [`docs/ABLATION_TEMPLATE.md`](ABLATION_TEMPLATE.md) for
  the chunk-size × threshold sweep that justifies the chosen defaults.

---

## 7. What we explicitly do not measure (and why)

- **Exact-string match against a reference answer**. References don't exist;
  writing them would inject our biases into the evaluation.
- **BLEU / ROUGE**. These reward surface similarity, not factual grounding,
  which is the actual axis of concern for a financial RAG.
- **End-to-end latency**. Nice to know but not relevant to the Milestone 5
  rubric. Included in the Streamlit demo for color but not as a pass/fail metric.
