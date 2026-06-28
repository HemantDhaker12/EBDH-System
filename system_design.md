# EDHC System Design Specification

This document provides a detailed overview of the system design, components, and data workflows of the **Evidence-Based Digital Hiring Committee (EDHC)** ranking pipeline.

---

## 1. System Architecture Diagram

The EDHC architecture is structured as a progressive funnel (**Three-Stage Sieve**) to process high-volume candidate inputs (100,000+ candidates) locally and efficiently on CPU-only workstations:

```
[100,000 Raw Candidate JSONL]
              │
              ▼ (Candidate Loading & Normalization)
[Normalized Candidate Profiles]
              │
              ▼ (Stage 1: Lexical BM25 pre-filtering)
   [Top 4,000 Lexical subset]
              │
              ▼ (Stage 2: Dense semantic search & RRF)
   [Top 2,000 Hybrid RRF subset]
              │
              ▼ (Stage 3: Feature Engineering & LTR Model)
    [39-Feature Matrix (2000x39)]
              │
              ▼ (LightGBM LambdaMART Inference)
    [Raw LTR Cohort Scores]
              │
              ▼ (Calibration & Custom Penalties)
    [Calibrated Candidates]
              │
              ▼ (Factual Reasoning Ledger)
[Top 100 Ranked Candidates (submission.csv)]
```

---

## 2. Core Subsystems and Components

### A. Candidate Normalization Subsystem
- **Responsibility**: Standardize raw candidate profiles (JSON/JSONL/JSONL.GZ) into a canonical structured model.
- **Key Modules**:
  - `CandidateProfileField`, `CareerHistoryField`, `SkillField`, `RedrobSignalsField`: Pydantic schemas validating fields, tenures, notice days, and expected salaries.
  - `normalize_candidate()`: Converts arbitrary schemas dynamically. It calculates leap-year adjusted tenure, resolves current roles, and compiles text blocks.

### B. Hybrid Retrieval Engine
- **Responsibility**: Reduce candidate pools while maintaining high recall.
- **Key Modules**:
  - `BM25Okapi`: Generates lexical scores over the tokenized profile text.
  - `SentenceTransformer`: Lazy loads the `all-MiniLM-L6-v2` encoder to compute dense vectors using E5-style query and passage prefixing.
  - `MockTransformer`: Fallback generator seeded with MD5 hashes to compute deterministic vectors if neural libraries are missing.
  - `Reciprocal Rank Fusion (RRF)`: Fuses lexical and semantic rankings using a constant $k=60$:
    $$RRF\_Score(c) = \frac{1}{60 + r_{BM25}(c)} + \frac{1}{60 + r_{dense}(c)}$$

### C. Feature Engineering Registry
- **Responsibility**: Extract and normalize numerical inputs for ranking models.
- **Key Modules**:
  - `FeatureGenerator`: Extracts **39 engineered features** categorized across retrieval, semantic similarities, career durations, technical specializations, notice/relocation availability, credibility warning counts, and quantitative achievement densities.
  - `CandidateFeatures`: Pydantic schema mapping the features into a flat `float32` array for inference.

### D. Learning-to-Rank (LTR) Model
- **Responsibility**: Score candidate relevance lists relative to each cohort.
- **Key Modules**:
  - `LambdaMARTRanker`: Integrates LightGBM `LGBMRanker` fit using the `lambdarank` objective and `ndcg` metric.
  - `LinearRanker`: Regression-based fallback predictor in case model weights are missing.

### E. Calibration and Tie-Breaking layer
- **Responsibility**: Apply business rules and ensure strict monotonic rankings.
- **Key Modules**:
  - `ScoreCalibrator`: Clamps normalized scores. Applies credibility penalties (timeline mismatches), consulting company ratio penalties, and long notice period overrides.
  - Deterministic tie-breaker: Sorts by score descending, then candidate ID ascending. Subtracts a tiny monotonic offset ($i \times 0.0001$) to guarantee unique, non-overlapping calibrated scores.

### F. Factual Reasoning Engine
- **Responsibility**: Construct objective, non-hallucinated explanations.
- **Key Modules**:
  - `ReasoningGenerator`: Compiles summaries using candidate titles, merged tenures, verified skills, and signals.
  - Deterministic variation: Uses the ASCII character sum of candidate IDs as a hash seed to rotate sentence structures and eliminate repetitive templates.

---

## 3. Data Workflows

### Inference Pipeline Workflow
1. **Load Pool**: `rank.py` streams candidates into `CandidateProfile` instances.
2. **Index BM25**: `HybridRetrievalEngine` tokenizes profile texts and indexes the corpus.
3. **Lexical Filtering**: BM25 scores the entire pool; the Top 4,000 candidates are filtered.
4. **Dense Semantic Retrieval**: The Top 4,000 candidates are encoded using SentenceTransformer and ranked.
5. **Fusion**: RRF merges lexical and semantic rankings, slicing the Top 2,000.
6. **Feature Matrix**: `FeatureGenerator` extracts 39 numerical features for the Top 2,000 candidates.
7. **LTR Score**: `LambdaMARTRanker` predicts raw scores.
8. **Calibrate & Penalize**: `ScoreCalibrator` applies consulting, availability, and credibility penalties.
9. **Tie-Break**: Calibrated records are sorted, and monotonic rank offsets are applied.
10. **Explain**: `ReasoningGenerator` generates summaries for the Top 100 candidates.
11. **Write Output**: Formats and saves the top list to `submission.csv`.

---

## 4. Key Design Specifications

- **Resource Limits**: Configured to run locally on CPU under 16GB RAM limits. High-recall pre-filtering ensures neural calculations are restricted to a small subset, finishing execution in under 3 minutes.
- **Zero API Dependencies**: All parses, validations, embeddings, and ranking steps execute completely locally. No candidate profile text is transmitted to external hosted API services.
- **Reproducibility Guarantee**: Fallback MD5 seeds and deterministic score offsets ensure identical candidate scores and rankings across runs.
