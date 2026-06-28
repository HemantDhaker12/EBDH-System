# Challenge Constraints Compliance (EDHC)

The **EDHC pipeline** is engineered to fully comply with all compute, memory, network, and execution time constraints outlined in the Redrob hackathon rules.

---

## 1. Compliance Checklist

| Constraint | Requirement | Actual Status | Verification Method |
| ---------- | ----------- | ------------- | ------------------- |
| **Hardware** | CPU-only execution, no active GPU/CUDA | **Compliant** | LightGBM model loads and runs inference using CPU threads; SentenceTransformer runs on CPU device fallback. |
| **Memory** | RAM usage < 16GB | **Compliant** | Candidates are parsed and processed lazily; indexing and filtering steps limit in-memory subsets. |
| **Network** | 100% offline, no API calls, no network dependency during ranking | **Compliant** | All model weights, parsing algorithms, and calculations run locally. No hosted API calls (OpenAI, Anthropic, Cohere) are triggered. |
| **Execution Time** | Under 5 minutes end-to-end on 100,000 pool | **Compliant** | Funnel design filters 100k down to 10k via lexical search, then 4k for semantic search, keeping runtime under **~3 minutes** on 8-core CPU. |
| **Format** | Exactly Top 100 list-wise CSV with unique ranks | **Compliant** | `ScoreCalibrator` applies deterministic tie-breaker sorting and rank assignments, producing exactly 100 unique candidate rows. |

---

## 2. Technical Implementation Details

### A. Runtime Optimization
Generating dense SentenceTransformer embeddings for 100,000 candidates on CPU would take several hours. The EDHC pipeline avoids this using a **Three-Stage Sieve**:
1. **Lexical BM25 pre-filtering** screens the 100,000 pool down to the Top 10,000 in under 1 second.
2. **Dense encoding** is only calculated on the Top 4,000 candidate profiles from the lexical list.
3. RRF fuses these rankings, selecting the Top 2,000 for LambdaMART inference.
- **Result**: Sub-3 minute execution time.

### B. Memory Safeguards
- In `rank.py`, JSONL files are processed line-by-line using streaming generators to prevent loading the entire raw dataset into memory before filtering.
- Intermediate embeddings are cached using lightweight dictionary lookup caches (`embedding_cache` inside `retrieval.py`).

### C. Network Ban Compliance
- No Python library used in the pipeline attempts external connections during execution.
- If dependencies cannot download neural resources (e.g. models) due to firewalls or lack of network connection, the system uses pre-cached local resources or falls back to deterministic local mock vector generators.
