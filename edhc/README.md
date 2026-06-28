# Evidence-Based Digital Hiring Committee (EDHC)

EDHC is a production-grade candidate ranking and verification system designed for high-precision, explainable candidate filtering against Job Descriptions. Rather than trusting raw resume keyword declarations, EDHC cross-references skill claims across job histories, projects, timelines, and credentials to build an auditable evidence ledger, score candidate features, rank them using Learning-to-Rank (LTR), and compile factual narratives free of LLM hallucinations.

---

## Technical Stack
* **Python**: 3.11+
* **Data Processing**: Pandas, Polars, NumPy
* **Search / Retrieval**: rank-bm25 (lexical), sentence-transformers (dense vector embeddings)
* **Learning-to-Rank**: LightGBM (LambdaMART), scikit-learn
* **APIs**: FastAPI, Uvicorn, Pydantic
* **Testing**: pytest

---

## Directory Structure Walkthrough

```
edhc/
├── app/
│   ├── config/          # Pydantic Settings and configurations
│   ├── schemas/         # Shared schemas (JD, Candidate, Evidence, Features)
│   ├── utils/           # Structured logging and disk cache tools
│   ├── preprocessing/   # Normalizer parses dates, cleans text and maps titles
│   ├── jd/              # Rubric parsing and competency extraction
│   ├── semantic/        # Hybrid BM25 & dense vector search index
│   ├── career/          # Experience calculation, trajectory and stability scores
│   ├── skills/          # Skills validator that cross-references durations
│   ├── behavior/        # Modifiers, notice period and availability scoring
│   ├── evidence/        # Computes multi-source alignment tokens
│   ├── consistency/     # Overlap, contradiction, and honeypot detection
│   ├── features/        # Feature vector assembly registry
│   ├── ranking/         # BaseRanker interface and LambdaMART model
│   ├── calibration/     # Min-max normalization, ties, and penalty mapping
│   ├── reasoning/       # Bulletproof, deterministic explanation ledger
│   ├── evaluation/      # Local audit metrics (NDCG, MAP, MRR, Precision)
│   ├── api/             # FastAPI REST endpoint routes
│   └── pipeline/        # Pipelines for LTR training and end-to-end inference
├── data/
│   ├── raw/             # Raw input JSON resume profiles
│   ├── processed/       # Canonical CandidateProfileNormalized files
│   ├── embeddings/      # Cached search index vector embeddings
│   └── caches/          # Local caches for intermediate scores
├── notebooks/           # Notebook workspace for model analysis
├── tests/               # Full pytest suite for all subsystems
├── scripts/             # Internal helper scripts
├── models/              # Serialized trained LTR models (.pkl)
├── outputs/             # Ranking reports (JSON/Markdown)
├── preprocess.py        # CLI to ingest and normalize raw candidates
├── train_ranker.py      # CLI to train the LambdaMART ranking model
├── rank.py              # CLI to score and rank candidates against a JD
├── requirements.txt     # Python package requirements
└── README.md            # Master documentation
```

---

## Setup & Ingest Instructions

### 1. Initialize and Activate Environment
To run the python pipeline, configure the local virtual environment `.venv` inside the project root:

On Windows (PowerShell):
```powershell
# In the EBDH workspace folder:
.venv\Scripts\Activate.ps1
```

### 2. Ingest & Preprocess Raw Profiles
To normalize candidate documents into standardized profiles:
```bash
python -m edhc.preprocess --generate-samples
```
This generates a sample raw JSON file in `data/raw/jane_doe_raw.json` (if empty) and normalizes it to `data/processed/<candidate_id>.json`.

### 3. Train the LambdaMART Ranker
To compile candidate feature profiles and fit the Learning-to-Rank model using Mock training pairs:
```bash
python -m edhc.train_ranker
```
This fits a LightGBM LambdaMART ranker and outputs the serialized model `models/lambdamart_model.pkl`.

### 4. Execute Candidate Ranking
To rank the normalized candidates against a Job Description:
```bash
python -m edhc.rank
```
This prints a clean ranking report to stdout and writes detailed outputs to:
* `data/outputs/ranking_report.json`
* `data/outputs/ranking_report.md` (complete with verified strengths, weaknesses, credibility scores, and hallucination-free explanations)

---

## Run Unit Tests
Verify all sub-components and mathematics (NDCG, Precision@K, Normalizer, Contradictions) using `pytest`:
```bash
pytest edhc/tests/
```
