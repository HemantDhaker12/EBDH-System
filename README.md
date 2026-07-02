# EDHC (Evidence-Based Digital Hiring Committee)

EDHC is an offline candidate discovery and ranking system developed for the **Redrob Candidate Discovery Challenge**.

The system retrieves relevant candidates using a hybrid lexical and semantic retrieval pipeline, ranks them using a LightGBM LambdaMART Learning-to-Rank model, applies deterministic score calibration, and generates factual reasoning for every shortlisted candidate.

The entire pipeline runs locally without external APIs and is designed to satisfy the hackathon's CPU-only execution constraints.

---

# Problem Statement

Traditional Applicant Tracking Systems (ATS) rely heavily on keyword matching, making them susceptible to keyword stuffing and poor semantic understanding.

The objective of this project is to identify and rank the best 100 candidates from a dataset of 100,000 candidate profiles by combining lexical retrieval, semantic retrieval, feature engineering, Learning-to-Rank, credibility analysis, and deterministic score calibration.

---

# System Architecture

```
                    100,000 Candidate Profiles
                              │
                              ▼
                 BM25 Lexical Retrieval
                   (Top 4,000 Candidates)
                              │
                              ▼
      Dense Semantic Retrieval (SentenceTransformer)
               + Reciprocal Rank Fusion (RRF)
                  (Top 2,000 Candidates)
                              │
                              ▼
                  39 Engineered Features
                              │
                              ▼
               LightGBM LambdaMART Ranker
                              │
                              ▼
                  Score Calibration
                              │
                              ▼
                Reasoning Generation
                              │
                              ▼
               Top 100 Ranked Candidates
                              │
                              ▼
                    submission.csv
```

The pipeline consists of three primary stages:

### 1. Candidate Normalization

Incoming candidate profiles are normalized into a consistent internal schema before downstream processing.

### 2. Hybrid Candidate Retrieval

Candidate discovery combines:

* BM25 lexical retrieval
* Dense semantic retrieval using the `all-MiniLM-L6-v2` SentenceTransformer model with E5-style query/document prefixing
* Reciprocal Rank Fusion (RRF) to merge lexical and semantic rankings

### 3. Learning-to-Rank

Retrieved candidates are represented using a 39-feature engineering pipeline and ranked using a LightGBM LambdaMART model.

The ranked candidates are then:

* calibrated,
* checked for credibility and consistency,
* assigned deterministic scores,
* accompanied by factual reasoning summaries.

---

# Repository Structure

```text
EBDH/
├── README.md
├── LICENSE
├── requirements.txt
├── submission_metadata.yaml
├── rank.py
├── main.py
├── submission.csv
├── docs/
│   ├── architecture.md
│   ├── retrieval_pipeline.md
│   ├── ranking_pipeline.md
│   ├── feature_engineering.md
│   ├── reasoning_engine.md
│   ├── training_pipeline.md
│   ├── challenge_constraints.md
│   └── design_decisions.md
├── edhc/
└── hackathon_assets/
```

---

# Requirements

* Python 3.11+
* Dependencies listed in `edhc/requirements.txt`
* Offline execution
* CPU-only execution
* Designed for systems with up to 16 GB RAM

---

# Quick Start

## Clone the Repository

```bash
git clone https://github.com/HemantDhaker12/EBDH-System
cd EBDH
```

## Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r edhc/requirements.txt
```

---

# Dataset

The repository does **not** include the official challenge dataset.

Download the official hackathon assets and place them inside:

```
hackathon_assets/
```

Required files:

* `candidates.jsonl`
* `validate_submission.py`

Optional:

* `sample_candidates.json` (used for sandbox demonstration)

---

# (Optional) Retrain the Model

Retraining is only required if the ranking model or training pipeline has been modified.

```bash
python edhc/train_ranker.py \
    --candidates ./hackathon_assets/candidates.jsonl \
    --sample-size 5000 \
    --retrieval-limit 3000
```
```bash
python edhc/train_ranker.py --candidates ./hackathon_assets/candidates.jsonl --sample-size 5000 --retrieval-limit 3000
```
---

# Reproduce submission.csv

Generate the final submission directly from the official candidate dataset.

```bash
python rank.py \
    --candidates ./hackathon_assets/candidates.jsonl \
    --out submission.csv
```

```bash
python rank.py --candidates ./hackathon_assets/candidates.jsonl --out submission.csv
```
No manual editing or post-processing is required.

---

# Validate the Submission

```bash
python hackathon_assets/validate_submission.py submission.csv
```

---

# Run Tests

```bash
pytest
```

---

# Project Highlights

* Hybrid BM25 + Semantic Retrieval
* Reciprocal Rank Fusion (RRF)
* LightGBM LambdaMART Learning-to-Rank
* 39 Engineered Features
* Candidate Credibility Analysis
* Deterministic Score Calibration
* Explainable Candidate Reasoning
* Fully Offline Execution
* CPU-Only Pipeline

---

# Challenge Constraints

The implementation is designed to satisfy the hackathon constraints:

* Offline execution
* CPU-only execution
* Deterministic ranking output
* No external API dependencies
* Designed to execute within the challenge runtime limit (≤5 minutes)

---



# Sandbox

A hosted sandbox demonstrating the ranking pipeline on a small candidate sample (≤100 candidates) will be added before submission.

**Sandbox URL**

```
https://huggingface.co/spaces/Hemant-space12/EBDH-system
```

---

# License

This project is licensed under the MIT License.
