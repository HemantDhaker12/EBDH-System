# Retrieval Pipeline (EDHC)

The **EDHC Retrieval Engine** uses a hybrid search approach (combining lexical and dense semantic signals) to select the Top 2,000 candidates from a candidate pool of 100,000. It is designed to run completely offline on CPU within strict runtime limits.

---

## 1. Pipeline Stages

The retrieval process consists of three main steps:

```
[100,000 Candidate Profiles]
            │
            ▼
┌───────────────────────────┐
│  1. Lexical Search (BM25) │  <-- Surfaces top 10,000 matches in <1s
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ 2. Dense Semantic Search  │  <-- Computes SentenceTransformer embeddings
└───────────┬───────────────┘      for the top 4,000 lexical candidates
            │
            ▼
┌───────────────────────────┐
│ 3. Reciprocal Rank Fusion │  <-- Combines rankings using RRF (k=60)
└───────────┬───────────────┘
            │
            ▼
[Top 2,000 Retrieval Subset]
```

### Stage 1: Lexical Search (BM25)
- **Engine**: The system builds a local `BM25Okapi` index (configured with $k_1 = 1.5, b = 0.75$).
- **Document Compilation**: For each candidate, a document text block is compiled containing:
  - Profile headline
  - Profile summary
  - Current title and industry
  - String-joined skills
  - Career history (job title, company, and descriptions)
  - Education (degree, field, and institution)
- **Tokenization**: Uses a regular expression `\b\w{2,}\b` to compile and tokenize lowercase terms.
- **Output**: Retrieves all candidate matches, ranks them, and slices the **Top 10,000 candidates** for the dense search stage.

### Stage 2: Dense Semantic Search (SentenceTransformer)
- **Scope**: To respect memory and execution time limits, dense encoding is restricted to a pre-filtered pool of size `min(len(candidate_ids), top_k * 2)` (which is the **Top 4,000 candidates** from lexical ranking).
- **Model**: Uses a local SentenceTransformer model (defaults to `all-MiniLM-L6-v2` in settings).
- **Formulation**:
  - Query prefix: `"query: {query_text}"`
  - Passage prefix: `"passage: {document_text}"`
- **Fallback Encoder**: If the `sentence_transformers` library cannot be loaded (e.g. package conflicts on host machine), the engine routes to a `MockTransformer`. This fallback computes the MD5 digest of each candidate's compiled profile text and uses it to seed a local random generator, producing stable, deterministic 384-dimensional representation vectors.
- **Output**: Computes dot-product cosine similarities between the query vector and candidate profile vectors to establish dense rankings for the candidate subset.

### Stage 3: Reciprocal Rank Fusion (RRF)
- **Algorithm**: Rfusion merges lexical and dense rankings without score scale dependency. The RRF scoring uses a constant factor $k=60$:
  $$RRF\_Score(c) = \frac{1}{60 + r_{BM25}(c)} + \frac{1}{60 + r_{dense}(c)}$$
  Where $r_{BM25}(c)$ is the lexical rank and $r_{dense}(c)$ is the dense rank. If a candidate is not present in the dense matching subset, their rank defaults to the pool size.
- **Final Output**: Slices the **Top 2,000 candidates** sorted by descending RRF scores to feed into the feature engineering and LambdaMART stages.
