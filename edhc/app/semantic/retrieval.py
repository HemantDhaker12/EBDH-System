import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi

from edhc.app.config.settings import settings
from edhc.app.schemas.candidate import CandidateProfile
from edhc.app.utils.logger import get_logger

logger = get_logger(__name__)

class HybridRetrievalEngine:
    """Manages lexical search (BM25) and dense semantic search, merging them using Reciprocal Rank Fusion (RRF)."""

    def __init__(self) -> None:
        self.candidates_db: Dict[str, CandidateProfile] = {}
        self.candidate_ids: List[str] = []
        self.bm25: Optional[BM25Okapi] = None
        self._model = None
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.diagnostics_cache: Dict[str, Dict[str, Any]] = {}

    @property
    def model(self):
        """Lazy load SentenceTransformer model when embedding on-the-fly is required."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model on-the-fly: {settings.EMBEDDING_MODEL_NAME}")
                self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            except Exception as e:
                logger.error(f"Failed to load sentence-transformers. Dense search fallback to mock. Error: {e}")
                class MockTransformer:
                    def encode(self, texts, **kwargs):
                        import hashlib
                        embs = []
                        for text in texts:
                            # Generate deterministic 384-dimensional representation using MD5 digest as random seed
                            h = hashlib.md5(text.encode('utf-8')).hexdigest()
                            seed_val = int(h[:8], 16)
                            rng = np.random.default_rng(seed_val)
                            embs.append(rng.standard_normal(384))
                        return np.vstack(embs)
                self._model = MockTransformer()
        return self._model

    def set_embedding_cache(self, cache: Dict[str, list]) -> None:
        """Inject precomputed embeddings to bypass neural computation on CPU."""
        logger.info(f"Injecting precomputed embedding cache containing {len(cache)} entries.")
        self.embedding_cache = {cid: np.array(vec, dtype=np.float32) for cid, vec in cache.items()}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 search."""
        return re.findall(r"\b\w{2,}\b", text.lower())

    def compile_document_text(self, candidate: CandidateProfile) -> str:
        """Compile a unified text block of the candidate profile for lexical and dense indexing."""
        p = candidate.profile
        skills_str = " ".join([s.name for s in candidate.skills])
        
        career_parts = []
        for exp in candidate.career_history:
            career_parts.append(f"{exp.title} at {exp.company} ({exp.description})")
        career_str = " ".join(career_parts)
        
        edu_parts = []
        for edu in candidate.education:
            edu_parts.append(f"{edu.degree} in {edu.field_of_study} at {edu.institution}")
        edu_str = " ".join(edu_parts)
        
        text_parts = [
            p.headline,
            p.summary,
            p.current_title,
            p.current_industry,
            skills_str,
            career_str,
            edu_str
        ]
        return " ".join([part for part in text_parts if part])

    def index(self, candidates: List[CandidateProfile]) -> None:
        """Build the lexical BM25 database and register profiles."""
        logger.info(f"Indexing {len(candidates)} candidates into lexical space...")
        self.candidates_db = {c.candidate_id: c for c in candidates}
        self.candidate_ids = [c.candidate_id for c in candidates]

        # Tokenize corpus for BM25
        tokenized_corpus = []
        for c in candidates:
            doc_text = self.compile_document_text(c)
            tokenized_corpus.append(self._tokenize(doc_text))

        if tokenized_corpus:
            self.bm25 = BM25Okapi(
                tokenized_corpus,
                k1=settings.BM25_K1,
                b=settings.BM25_B
            )
            logger.info("Lexical BM25 index built successfully.")
        else:
            self.bm25 = None
            logger.warning("BM25 corpus was empty. No indexing performed.")

    def search(self, query: str, top_k: int = 4000) -> List[Tuple[str, float]]:
        """Perform Reciprocal Rank Fusion (RRF) search and return top_k candidate IDs and RRF scores.

        Fuses lexical (BM25) and dense (SentenceTransformer) rankings using RRF (k=60).
        """
        if not self.candidate_ids:
            logger.warning("Search called on an empty retrieval index.")
            return []

        logger.info(f"Starting hybrid retrieval for query: '{query}' with Top-K={top_k} limit.")

        # 1. Lexical Search (BM25)
        bm25_ranks: Dict[str, int] = {}
        bm25_scores_raw: Dict[str, float] = {}
        sorted_cids_bm25 = []
        if self.bm25:
            query_tokens = self._tokenize(query)
            bm25_scores = np.array(self.bm25.get_scores(query_tokens))
            
            # Find indices and sort
            sorted_indices = np.argsort(bm25_scores)[::-1]
            for rank, idx in enumerate(sorted_indices, 1):
                cid = self.candidate_ids[idx]
                bm25_ranks[cid] = rank
                bm25_scores_raw[cid] = float(bm25_scores[idx])
                sorted_cids_bm25.append(cid)
        else:
            # Fallback uniform rank
            for rank, cid in enumerate(self.candidate_ids, 1):
                bm25_ranks[cid] = rank
                bm25_scores_raw[cid] = 0.0
                sorted_cids_bm25.append(cid)

        # Pre-filter: take the top top_k * 2 candidates for semantic dense matching
        # to ensure fast CPU processing time.
        match_pool_size = min(len(self.candidate_ids), top_k * 2)
        filter_candidates = sorted_cids_bm25[:match_pool_size]

        # 2. Dense Semantic Search (E5) on pre-filtered candidates
        dense_ranks: Dict[str, int] = {}
        dense_scores_raw: Dict[str, float] = {}
        try:
            # Generate query embedding
            query_emb = self.model.encode([f"query: {query}"])[0]
            query_emb_norm = query_emb / np.linalg.norm(query_emb)
            
            # Collect corpus embeddings (lookup cache or encode on-the-fly)
            missing_cids = []
            corpus_embs = []
            
            for cid in filter_candidates:
                if cid in self.embedding_cache:
                    corpus_embs.append(self.embedding_cache[cid])
                else:
                    missing_cids.append(cid)
                    corpus_embs.append(None)
                    
            if missing_cids:
                logger.info(f"Calculating embeddings on-the-fly for {len(missing_cids)} missing candidates in filtered pool.")
                missing_texts = [f"passage: {self.compile_document_text(self.candidates_db[cid])}" for cid in missing_cids]
                missing_embs = self.model.encode(missing_texts, show_progress_bar=False)
                
                # Write back to corpus list and update cache
                m_ptr = 0
                for i, cid in enumerate(filter_candidates):
                    if corpus_embs[i] is None:
                        vec = missing_embs[m_ptr]
                        corpus_embs[i] = vec
                        # Store in active cache
                        self.embedding_cache[cid] = vec
                        m_ptr += 1

            # Stack and normalize embeddings
            corpus_embs_stacked = np.vstack(corpus_embs)
            corpus_embs_norm = corpus_embs_stacked / np.linalg.norm(corpus_embs_stacked, axis=1, keepdims=True)
            
            # Cosine similarity dot product
            cosine_similarities = np.dot(corpus_embs_norm, query_emb_norm)
            
            # Sort and rank within pool
            sorted_dense_pool_indices = np.argsort(cosine_similarities)[::-1]
            for rank, idx in enumerate(sorted_dense_pool_indices, 1):
                cid = filter_candidates[idx]
                dense_ranks[cid] = rank
                dense_scores_raw[cid] = float(cosine_similarities[idx])
        except Exception as e:
            logger.error(f"Dense semantic search failed, using lexical rankings as fallback. Error: {e}")
            for rank, cid in enumerate(filter_candidates, 1):
                dense_ranks[cid] = rank
                dense_scores_raw[cid] = 0.0

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF Score(c) = 1 / (60 + r_BM25(c)) + 1 / (60 + r_dense(c))
        k = 60
        rrf_results = []
        diagnostics = {}
        
        for cid in filter_candidates:
            r_bm25 = bm25_ranks.get(cid, len(self.candidate_ids))
            r_dense = dense_ranks.get(cid, len(self.candidate_ids))
            
            score_rrf = (1.0 / (k + r_bm25)) + (1.0 / (k + r_dense))
            rrf_results.append((cid, score_rrf))
            
            # Store diagnostics
            diagnostics[cid] = {
                "bm25_score": bm25_scores_raw.get(cid, 0.0),
                "bm25_rank": r_bm25,
                "dense_similarity": dense_scores_raw.get(cid, 0.0),
                "dense_rank": r_dense,
                "rrf_score": score_rrf
            }

        # Sort by RRF score descending
        rrf_results.sort(key=lambda x: x[1], reverse=True)
        top_k_results = rrf_results[:top_k]

        # Annotate rank in diagnostics and persist
        final_diagnostics = {}
        for rank, (cid, score) in enumerate(top_k_results, 1):
            diag = diagnostics[cid]
            diag["rrf_rank"] = rank
            final_diagnostics[cid] = diag
            
        self.diagnostics_cache = final_diagnostics
        self._persist_diagnostics()

        logger.info(f"Hybrid RRF completed. Surfaced {len(top_k_results)} candidates.")
        return top_k_results

    def _persist_diagnostics(self) -> None:
        """Save intermediate retrieval rankings and metrics to disk for evaluation/debugging."""
        try:
            settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            diagnostics_file = settings.CACHE_DIR / "retrieval_diagnostics.json"
            with open(diagnostics_file, "w", encoding="utf-8") as f:
                json.dump(self.diagnostics_cache, f, indent=4)
            logger.info(f"Persisted intermediate retrieval diagnostics to {diagnostics_file}")
        except Exception as e:
            logger.error(f"Failed to write retrieval diagnostics to disk: {e}")
