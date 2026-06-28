# EDHC (Evidence-Based Digital Hiring Committee)

The Evidence-Based Digital Hiring Committee (EDHC) is a candidate search and ranking pipeline developed for the Redrob Candidate Discovery Challenge. The system screens and ranks candidate profiles against target job description rubrics using hybrid search models and a list-wise Learning-to-Rank framework.

EDHC leverages multi-stage pre-filtering to process 100,000 candidate profiles locally on CPU, ensuring compliance with strict compute constraints. The system identifies profile chronology conflicts (honeypots), performs score calibration, and generates dynamic factual reasoning justifications for each final candidate.

## Problem Statement

ATS platforms often rely on simple keyword matching, which is vulnerable to keyword stuffing and cannot detect timeline anomalies. The challenge is to identify and rank the Top 100 candidates from a pool of 100,000 profiles against a job description rubric, ignoring keyword stuffers, penalizing inconsistent timelines, and generating objective justifications.

## System Overview

```
[100,000 Candidates Pool]
         │
         ▼ (Stage 1: Lexical pre-filter via BM25 Okapi)
[Top 10,000 Candidates]
         │
         ▼ (Stage 2: Dense semantic search via E5/MiniLM + RRF)
[Top 2,000 Candidates]
         │
         ▼ (Stage 3: 39-feature matrix + LightGBM LambdaMART ranker)
[Top 100 Calibrated Candidates] ──► Write to submission.csv
```

The pipeline operates in three stages:
* **Candidate normalization**: Dynamic conversion of candidate profile inputs into standardized structures.
* **Hybrid retrieval**: Keyword matching (BM25) combined with dense semantic similarity (E5 prefixing via `all-MiniLM-L6-v2` embeddings) fused via Reciprocal Rank Fusion (RRF, $k=60$).
* **LambdaMART ranking + score calibration + reasoning generation**: Cohort ranking via a LightGBM LambdaMART model, score adjustment for notice periods and consulting backgrounds, tie-breakers, and narrative committee justifications.

## Repository Structure

```
EBDH/
├── LICENSE                        # MIT License
├── README.md                      # Primary project documentation
├── rank.py                        # Root candidate ranking CLI orchestrator
├── hackathon_assets/              # Candidate schemas, validators, and sample inputs
├── docs/                          # Core implementation details
│   ├── architecture.md
│   ├── retrieval_pipeline.md
│   ├── ranking_pipeline.md
│   ├── feature_engineering.md
│   ├── reasoning_engine.md
│   ├── training_pipeline.md
│   ├── challenge_constraints.md
│   └── design_decisions.md
└── edhc/                          # Core application Python package
```

## Requirements

* **Python version**: Python 3.11
* **Dependencies**: Specified in `edhc/requirements.txt`
* **Offline execution**: 100% local (no external API calls or network requests)
* **CPU-only**: Designed and optimized for CPU hosts with ≤16GB RAM

# Quick Start

## 1. Clone the repository

```bash
git clone <repo_url>
cd EBDH
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

## 3. Activate it

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r edhc/requirements.txt
```

## 5. (Optional) Retrain the model

```bash
python edhc/train_ranker.py --candidates ./hackathon_assets/candidates.jsonl --sample-size 5000 --retrieval-limit 3000
```

## 6. Generate submission.csv

```bash
python rank.py --candidates ./hackathon_assets/candidates.jsonl --out submission.csv
```

## 7. Validate the submission

```bash
python hackathon_assets/validate_submission.py submission.csv
```

## 8. Run tests

```bash
pytest
```

## Dataset

The repository does not include the full candidate dataset. 

Download the official hackathon assets and place them inside:
`hackathon_assets/`

Required files:
- `candidates.jsonl`
- `validate_submission.py`
- `sample_candidates.json` (optional for sandbox/demo)

## Project Highlights

* Hybrid BM25 + Semantic Retrieval
* Reciprocal Rank Fusion
* LambdaMART Learning-to-Rank
* 39 engineered features
* Explainable ranking
* Offline execution
* CPU-only

## Constraints

* Offline execution
* CPU only
* Under hackathon runtime limit (under ~3 minutes execution)
* Deterministic output

## Sandbox

* **Sandbox URL Placeholder**: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SANDBOX_PLACEHOLDER`
* Demonstrates pipeline execution on `sample_candidates.json` (≤100 candidates) and outputs a ranked CSV.

## License

This repository is licensed under the [MIT License](LICENSE).
