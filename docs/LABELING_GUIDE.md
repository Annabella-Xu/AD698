# Labeling Guide — Human Ground Truth for Hit@k

## Why this matters

The first draft of `labeled_pairs_template.xlsx` auto-populated `gold_chunk_id_contains`
from the first chunk the system itself returned. Evaluating the system against
that reference is circular — it measures whether the system ranks highly the
chunk it already ranked first, not whether retrieval is actually correct.

**Fix**: human-label the gold chunks by reading the actual 10-K text. This
guide is the protocol.

## Target

- **25 labeled pairs** across the 20 domain questions (5 bonus for statistical margin).
- **≥ 5 questions per Item** (6, 7, 7A, 8) so we can compute per-Item Hit@k.
- **2 labelers per question** ideally — one labels, one verifies. If only one
  labeler, flag disagreements in the `notes` column.

## Protocol

### Step 1 — Run Milestones 1–2

```bash
python -m src.rag build --filings-dir data/filings --cache-dir .cache
```

This populates `.cache/chunks.jsonl`. Each line is a chunk record with
`chunk_id`, `item`, `company`, and `text`.

### Step 2 — For each row in `data/labeled_pairs.csv`

1. Open `labeled_pairs.csv`. Find a row with `TO LABEL` in the notes column.
2. Read the `question` and `labeling_hint`. The hint describes the phrase
   or subheading to search for.
3. **Pick a firm** from your corpus that has the material discussed.
   - Some firms won't have all disclosures (e.g., not every firm has
     goodwill impairment). The hint flags cases where firm choice matters.
4. **Find the chunk** by searching `.cache/chunks.jsonl` for the firm's
   chunks in the gold Item, and reading the `text` field:

   ```bash
   jq -c 'select(.item=="Item 7" and .company=="APPLE INC")' .cache/chunks.jsonl \
     | jq -r '[.chunk_id, .text[0:200]] | @tsv'
   ```

   Or open `.cache/chunks.jsonl` in a pandas notebook.

5. Select the chunk whose text **most directly answers** the question. Prefer
   a chunk whose first sentence answers the question; avoid chunks that only
   tangentially touch the topic.
6. Paste the `chunk_id` into `gold_chunk_id_contains`. You can paste the full
   id (exact match) or a distinctive substring (e.g., the firm ticker +
   index — `AAPL-Item7-012`) so the label survives minor re-chunking.
7. Paste the firm name into `gold_company`.
8. Replace `TO LABEL` in `notes` with a short justification (one sentence).

### Step 3 — Verify

Run evaluation:

```bash
python -m src.rag eval --labeled-pairs data/labeled_pairs.csv --verbose
```

Read the per-question output. For misses (✗), open the `top1_chunk` text
and confirm the miss is a real miss, not a labeling error.

## Common pitfalls

- **Picking too-narrow a chunk**. A chunk that only says "see Note 6" is not
  a good gold — choose the chunk with the actual disclosure.
- **Picking a TOC chunk**. If your chunk text looks like `Item 7 ........ 42`,
  that's a TOC bleed that survived `filter_toc_hits`. Don't use it as gold
  — fix the cleaning instead.
- **Relying on one firm**. Spread labels across 5+ firms so Hit@k isn't
  dominated by any single filing's quirks.

## Baseline to compare against

Before reporting Hit@k, also run a **random-Item baseline** to confirm
the score is meaningful:

```python
from src.rag import retrieve, build_index
import random, pandas as pd
index = build_index("data/filings", ".cache")
df = pd.read_csv("data/labeled_pairs.csv").dropna(subset=["gold_chunk_id_contains"])
random.seed(0)
all_items = index.cfg.allowed_items
hits = 0
for _, r in df.iterrows():
    wrong_item = random.choice([i for i in all_items if i != r.gold_item])
    got = retrieve(index, r.question, allowed_items=[wrong_item], k=5)
    if any(r.gold_chunk_id_contains in h["chunk_id"] for h in got):
        hits += 1
print(f"Random-Item baseline Hit@5 = {hits/len(df):.1%}")
```

Typical baseline is ~5–15%. **Your system's Hit@5 should be ≥ 4× this.**
