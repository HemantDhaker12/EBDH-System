# Architectural Design Decisions (EDHC)

This document explains the rationale behind the key architectural choices, algorithmic decisions, and constraint strategies implemented in the EDHC project.

---

## 1. Rationale Behind Core Engineering Decisions

### A. Why BM25 Before Dense Retrieval (Two-Tier Retrieval Sieve)
- **Problem**: Generating dense E5 SentenceTransformer embeddings on CPU for 100,000 candidates takes several hours.
- **Solution**: We employ a fast, low-overhead lexical search step using BM25 (`rank_bm25` library) first.
- **Design Decision**: Lexical search builds a BM25 index on compiled profile text, screening 100,000 candidates down to 10,000 in under 1 second. Dense neural encoding is then calculated on-the-fly *only* for the top 4,000 candidate profiles.
- **Benefit**: This two-tier funnel reduces neural computation latency by over 95%, satisfying the 5-minute hackathon execution constraint on CPU.

### B. Why Reciprocal Rank Fusion (RRF)
- **Problem**: Lexical similarity (BM25) and dense semantic similarity (E5 cosine distances) produce scores on completely different scales, making direct weighted score summation unstable.
- **Solution**: We use Reciprocal Rank Fusion (RRF) to merge the rankings.
- **Design Decision**: RRF with a standard constant $k=60$ fuses lexical ranks and dense semantic ranks without parameter tuning or scaling issues.
- **Benefit**: Ensures a robust, high-recall retrieval subset of 2,000 candidates that includes both exact keyword matches and contextually relevant synonyms.

### C. Why Learning-to-Rank (LambdaMART)
- **Problem**: Classic pointwise classification/regression models predict scores independently for each candidate, ignoring the relative list-wise context of a cohort.
- **Solution**: We train a list-wise ranking model using LightGBM LambdaMART.
- **Design Decision**: LambdaMART (`objective="lambdarank"`, `metric="ndcg"`) optimizes NDCG directly by modeling the pairwise differences in candidate relevance.
- **Benefit**: Fits list-wise ranking evaluation metrics exactly and optimizes candidate ordering over query-specific groups.

### D. Why Evidence Verification Exists
- **Problem**: Candidates frequently engage in "keyword stuffing" (declaring skills in a flat list without ever using them in professional work or projects).
- **Solution**: We implement the `EvidenceVerifier`.
- **Design Decision**: The verifier searches for occurrences of declared competencies inside other independent profile sections (experience descriptions, summary, and project details) and computes an `evidence_strength` metric.
- **Benefit**: Effectively filters out keyword-stuffer profiles while highlighting candidates with proven, contextual exposure in their career timelines.

### E. Why Credibility is a Soft Feature (with Hard Calibration Override)
- **Problem**: Hard-disqualifying candidates during early funneling stages can accidentally discard borderline candidates with minor profile noise.
- **Solution**: We model credibility as a soft feature in the LambdaMART matrix, backed by a severe calibration override for major anomalies.
- **Design Decision**:
  - `credibility_score` is a continuous feature $[0.0, 1.0]$ in the feature matrix, allowing LambdaMART to learn its relative importance.
  - If the profile triggers severe honeypot indicators (`credibility_score < 0.05`), the `ScoreCalibrator` applies a massive override penalty of **`-2.0`**, forcing their final score down to `0.0000`.
- **Benefit**: Maintains a flexible modeling pipeline while guaranteeing that dishonest honeypots are pushed to the bottom of the final submission list.

### F. Why Hard Filters Were Removed
- **Design Decision**: Traditional ATS systems use hard Boolean filters (e.g., must have 5+ years experience, must be immediately available). This approach misses stellar candidates who fall just short of one threshold.
- **Benefit**: By removing hard filters and substituting them with continuous feature scores (notice period scores, scaled YOE features) and soft calibration penalties, the system evaluates candidates holistically.

### G. Why the Pipeline is Fully Offline
- **Design Decision**: The pipeline depends 100% on local components and contains no hosted API endpoints (such as OpenAI GPT-4 or Anthropic Claude API).
- **Benefit**: Satisfies security restrictions (no candidate data is leaked to external APIs), guarantees deterministic reproducibility, and complies with network-free execution environments.
