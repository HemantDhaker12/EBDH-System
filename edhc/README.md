# EDHC Python Package

This directory contains the core implementation of the **Evidence-Based Digital Hiring Committee (EDHC)** candidate discovery and ranking system.

Unlike the repository root `README.md`, which explains how to reproduce the hackathon submission, this document focuses on the internal package architecture and developer workflow.

---

# Technology Stack

| Component          | Technology            |
| ------------------ | --------------------- |
| Language           | Python 3.11+          |
| Data Processing    | NumPy, Pandas, Polars |
| Lexical Retrieval  | rank-bm25             |
| Semantic Retrieval | sentence-transformers |
| Learning-to-Rank   | LightGBM (LambdaMART) |
| API                | FastAPI, Uvicorn      |
| Validation         | Pydantic              |
| Testing            | pytest                |

---

# Package Structure

```text
edhc/
├── app/
│   ├── api/                # FastAPI routes
│   ├── behavior/           # Notice period & availability analysis
│   ├── calibration/        # Score calibration & tie handling
│   ├── career/             # Career trajectory analysis
│   ├── config/             # Project configuration
│   ├── consistency/        # Timeline validation & credibility checks
│   ├── evaluation/         # Ranking evaluation utilities
│   ├── evidence/           # Multi-source evidence verification
│   ├── features/           # Feature generation
│   ├── jd/                 # Job description parsing
│   ├── pipeline/           # Training & inference pipelines
│   ├── preprocessing/      # Candidate normalization
│   ├── ranking/            # LambdaMART ranker
│   ├── reasoning/          # Candidate reasoning generation
│   ├── schemas/            # Shared Pydantic models
│   ├── semantic/           # BM25, dense retrieval & RRF
│   ├── skills/             # Skill verification
│   └── utils/              # Logging & caching
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── embeddings/
│   └── caches/
│
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── tests/
│
├── preprocess.py
├── train_ranker.py
├── rank.py
├── requirements.txt
└── README.md
```

---

# Package Workflow

The package consists of four major stages.

## 1. Candidate Normalization

```text
Raw Candidate
      │
      ▼
Candidate Normalizer
      │
      ▼
Canonical Candidate Schema
```

Responsibilities:

* Normalize profile structure
* Parse dates
* Standardize titles
* Compute experience

---

## 2. Candidate Retrieval

```text
Candidates
      │
      ▼
BM25 Retrieval
      │
      ▼
Dense Semantic Retrieval
      │
      ▼
Reciprocal Rank Fusion (RRF)
```

Responsibilities:

* Lexical retrieval
* Semantic similarity search
* Candidate fusion

---

## 3. Feature Engineering

Each retrieved candidate is transformed into the feature representation consumed by the Learning-to-Rank model.

Feature groups include:

* Career progression
* Experience
* Domain relevance
* Company history
* Skill verification
* Notice period
* Credibility
* Evidence alignment

---

## 4. Candidate Ranking

```text
Feature Matrix
      │
      ▼
LightGBM LambdaMART
      │
      ▼
Score Calibration
      │
      ▼
Reasoning Generation
```

Responsibilities:

* Predict ranking scores
* Apply deterministic calibration
* Generate factual candidate summaries

---

# Internal CLI Commands

The package provides three command-line utilities.

## Preprocess Candidates

```bash
python -m edhc.preprocess --generate-samples
```

Normalizes raw candidate profiles into the canonical schema.

---

## Train the Ranker

```bash
python -m edhc.train_ranker
```

Trains the LambdaMART Learning-to-Rank model and stores the serialized model in:

```text
models/lambdamart_model.pkl
```

---

## Run Package Ranking

```bash
python -m edhc.rank
```

Runs the internal ranking pipeline and generates diagnostic outputs.

> **Note:** The official hackathon submission is generated using the repository root `rank.py`. Refer to the root `README.md` for submission reproduction instructions.

---

# Running Tests

Execute the complete unit test suite:

```bash
pytest edhc/tests/
```

---

# Developer Notes

* The package is fully offline.
* No external APIs are required during inference.
* Ranking is deterministic for identical inputs.
* Individual modules can be tested independently through the `tests/` directory.
* Technical implementation details are documented in the repository's `docs/` directory.
