# Ablation Template — Chunk Size × Similarity Threshold

> **Add to the design brief as §11**, or as an appendix to §5 (Retrieval Risks).
> Justifies `CHUNK_TOKENS=500` and `MIN_SIM=0.20` with data instead of intuition.

---

## Experimental grid

Re-run Milestones 2–5 for each cell below. The notebook cell §5.5 runs
the sweep automatically when `ABLATION_MODE = True`.

|  | MIN_SIM = 0.10 | MIN_SIM = 0.20 | MIN_SIM = 0.30 |
|---|---|---|---|
| **CHUNK_TOKENS = 256** | Hit@5 = __ · Refusal = __ | Hit@5 = __ · Refusal = __ | Hit@5 = __ · Refusal = __ |
| **CHUNK_TOKENS = 500** (default) | Hit@5 = __ · Refusal = __ | Hit@5 = __ · Refusal = __ | Hit@5 = __ · Refusal = __ |
| **CHUNK_TOKENS = 1000** | Hit@5 = __ · Refusal = __ | Hit@5 = __ · Refusal = __ | Hit@5 = __ · Refusal = __ |

*Rows held constant: `CHUNK_OVERLAP = 0.1 × CHUNK_TOKENS`, all other config
defaults from `src/rag.py:Config`.*

---

## How to read the results

- **Hit@5** should peak at a mid-range chunk size. Very small chunks
  fragment the answer across multiple chunks (recall drops); very large
  chunks dilute the answer within off-topic text (also drops recall).
- **Refusal rate** rises with `MIN_SIM`. At `MIN_SIM = 0.30` the system
  should refuse many valid questions; at `MIN_SIM = 0.10` it should
  accept too many marginal ones (hallucination risk).
- **Chosen defaults** — fill in after running: we choose the cell that
  maximizes Hit@5 subject to refusal rate < 30%.

## Fill-in summary sentence (for the brief)

> Based on the sweep, we chose **CHUNK_TOKENS = 500** and **MIN_SIM = 0.20**.
> This configuration achieved Hit@5 = __% with a refusal rate of __%;
> alternatives within ±0.05 were within the noise margin, while
> more aggressive settings (e.g., MIN_SIM = 0.30) reduced Hit@5 to __%.
