# Ranking Pipeline (EDHC)

The **EDHC Ranking Engine** uses a LightGBM LambdaMART machine learning ranker combined with a post-processing calibration layer. This structure ensures that candidates are ranked by their matching features, while enforcing hard/soft business penalties and guaranteeing deterministic tie-breaks.

---

## 1. Pipeline Flow

```
[Top 2,000 Retrieved Candidates]
               │
               ▼
┌───────────────────────────────┐
│     1. Feature Generation     │  <-- Extracts 39 numerical features
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│      2. LambdaMART Model      │  <-- Predicts raw list-wise relevance
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│     3. Score Calibration      │  <-- Normalizes raw scores & applies:
└──────────────┬────────────────┘      - Credibility/Honeypot penalties
               │                       - Consulting background penalties
               │                       - Notice period penalties
               ▼
┌───────────────────────────────┐
│  4. Deterministic Tie-Breaker │  <-- Sorts by ID, applies minor offset
└──────────────┬────────────────┘      to output unique calibrated scores
               │
               ▼
[Exactly Top 100 final list]
```

---

## 2. LambdaMART Model

- **Algorithm**: LightGBM `LGBMRanker` configured with `objective="lambdarank"` and `metric="ndcg"`.
- **Hyperparameters**:
  - `boosting_type`: `"gbdt"` (Gradient Boosted Decision Trees)
  - `learning_rate`: `0.05`
  - `num_leaves`: `31`
  - `n_estimators`: `100`
- **Feature Matrix**: A flat matrix of shape (2000, 39) containing the numerical representation of each candidate.
- **Inference fallback**: If the serialized model (`lambdamart_model.pkl`) is missing, the engine falls back to a heuristic weighted combination scoring function:
  $$Score = 0.30 \cdot rrf\_score + 0.25 \cdot domain\_score + 0.15 \cdot evidence\_score + \dots$$

---

## 3. Score Calibration & Penalties

To align rankings with strategic hiring constraints (like discouraging honeypot traps, long notice periods, and pure consulting backgrounds), candidate scores are processed by a `ScoreCalibrator`:

1. **Min-Max Normalization**: Raw LambdaMART predictions are scaled to $[0.0, 1.0]$.
2. **Business Penalties**:
   - **Severe Honeypot Penalty**: If `credibility_score < 0.05` (indicating timeline contradictions, expert skill overlaps, or salary anomalies), the score is penalized by **`-2.0`**.
   - **Low Credibility Penalty**: If `credibility_score <= 0.7` or `contradiction_count > 0`, the score is penalized by **`-0.15`**.
   - **Services Company Penalty**: Pure services profiles (`services_company_ratio >= 0.99`) receive a penalty of **`-0.25`**, while high-services profiles ($> 0.5$) receive a penalty of **`-0.10`**.
   - **Long Notice Period Penalty**: Notice periods exceeding 60-90 days (`notice_period_score <= 0.2`) receive a penalty of **`-0.10`**.
3. **Clamping**: The score is clamped to ensure it lies in $[0.0, 1.0]$.

---

## 4. Deterministic Tie-Breaking & Sorting

- **Sorting**: Candidates are sorted by calibrated score descending, and then alphabetically by `candidate_id` ascending.
- **Tie-Breaking Offset**: To prevent identical scores (bucketization) in the final 4-decimal CSV output, a strictly monotonic decreasing offset is subtracted based on the candidate's rank:
  $$Score_{calibrated} = \max\left(0.0, \text{round}\left(Score_{clamped} - (i \times 0.0001), 4\right)\right)$$
  Where $i$ is the 0-indexed position in the final sorted array.
- **Result**: Ensures that all scores in the top 100 are unique and monotonically decreasing.
