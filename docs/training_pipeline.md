# Training Pipeline (EDHC)

The LTR training pipeline (`train_ranker.py`) is designed to optimize a local LightGBM LambdaMART ranking model (`lambdamart_model.pkl`). The model learns to prioritize true Senior AI/ML candidates while penalizing entry-level profiles, irrelevant roles (sales, marketing), and honeypot profiles.

---

## 1. Funnel Flow

```
[Candidate JSONL Pool]
          │
          ▼
┌───────────────────────────────┐
│     1. Query Categorization    │  <-- ml, search, backend, data
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│     2. Hybrid Retrieval Pool   │  <-- BM25 + Dense semantic retrieval
└──────────────┬────────────────┘      (Slices top 3,000 candidates per query)
               │
               ▼
┌───────────────────────────────┐
│  3. Multi-Dimensional Labels  │  <-- Computes relevance labels (0 to 4)
└──────────────┬────────────────┘      incorporating specialty and experience
               │
               ▼
┌───────────────────────────────┐
│     4. Contiguous Grouping    │  <-- Sorts by Query ID for LightGBM
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│      5. Validation Splits     │  <-- Splits dataset 80/20 per group
└──────────────┬────────────────┘
               │
               ▼
[Trained lambdamart_model.pkl]
```

---

## 2. Multi-Dimensional Relevance Labels

relevance labels are computed using a continuous multi-dimensional formula mapped to integers $[0, 4]$:

- **Specialty Alignment**: Adds points based on JD title Jaccard overlaps, dense semantic similarities, and domain relevance keywords.
- **Experience (YOE) Scaling**:
  - Rewards candidates in the target sweet-spot: **`5 to 15 years`** receives a boost.
  - Penalizes entry-level profiles ($< 2.5$ years).
  - Penalizes overqualified profiles ($> 20$ years).
- **Seniority & Progression**: Boosts candidates with leadership keyword triggers (e.g. Lead, Principal) and positive promotion velocity.
- **Competency & Evidence**: Boosts candidates with high competency coverages and cross-source corroborated evidence strengths.
- **Irrelevant Background Penalties**: Deducts points for keywords related to design, marketing, content, sales, or HR.
- **Honeypot Disqualification**: Overrides final labels to **`0`** if credibility scores are $< 0.05$.

---

## 3. Data Alignment & Sorting

- **Contiguous Sorting**: LightGBM requires all samples for a specific query group to be contiguous in the training array. The training pipeline sorts arrays by Query ID before fitting.
- **Group Array**: Computes `group` parameter array representing the size of each contiguous query chunk.

---

## 4. Validation & Checks

The training script runs several validation steps:
- **Validation Split**: Splitting 80/20 per group.
- **Dimension Check**: Asserts that all computed feature vectors have exactly 39 features.
- **Value Assertions**: Verifies that features contain no NaNs or infinite values.
- **Variance Check**: Asserts that predicted validation scores have a variance $> 1e-4$ to prevent score collapse.
- **NDCG Metrics**: Reports NDCG@1, NDCG@3, and NDCG@5 metrics.
- **Pairwise Ordering Accuracy**: Measures percent of correctly ordered pairs on validation sets (confirming accuracy $> 95\%$).
