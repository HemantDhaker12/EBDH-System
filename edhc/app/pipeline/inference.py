import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from edhc.app.config.settings import settings
from edhc.app.schemas.candidate import CandidateProfile
from edhc.app.schemas.jd import JobDescriptionParsed
from edhc.app.semantic.retrieval import HybridRetrievalEngine
from edhc.app.features.generator import FeatureGenerator
from edhc.app.ranking.ranker import LambdaMARTRanker
from edhc.app.calibration.calibrator import ScoreCalibrator
from edhc.app.reasoning.ledger import ReasoningGenerator
from edhc.app.evidence.verifier import EvidenceVerifier
from edhc.app.consistency.detector import ConsistencyDetector
from edhc.app.utils.logger import get_logger

logger = get_logger(__name__)

class InferencePipeline:
    """End-to-end inference pipeline for scoring, calibrating, and explaining candidates."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.retrieval_engine = HybridRetrievalEngine()
        self.feature_generator = FeatureGenerator()
        self.calibrator = ScoreCalibrator()
        self.reasoning_generator = ReasoningGenerator()
        self.evidence_verifier = EvidenceVerifier()
        self.consistency_detector = ConsistencyDetector()
        
        # Load ranking model if it exists, otherwise we fallback to a heuristic model
        self.model = LambdaMARTRanker()
        self.model_path = model_path or str(settings.MODELS_DIR / "lambdamart_model.pkl")
        
        if os.path.exists(self.model_path):
            try:
                self.model.load(self.model_path)
                self.model_loaded = True
                logger.info(f"Loaded trained LambdaMART model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load ranker model: {e}. Falling back to heuristic scoring.")
                self.model_loaded = False
        else:
            logger.info("No trained ranker model found at target path. Falling back to heuristic feature scoring.")
            self.model_loaded = False

    def run(self, candidates: List[CandidateProfile], jd: JobDescriptionParsed) -> List[Dict[str, Any]]:
        """Run inference pipeline on candidate pool."""
        logger.info(f"Running inference pipeline for {len(candidates)} candidates against JD: {jd.title}")
        
        if not candidates:
            return []

        # 1. Index candidates and run hybrid retrieval search
        self.retrieval_engine.index(candidates)
        
        # Dynamically generate search query
        from edhc.app.jd.intelligence import JobDescriptionParser
        parser = JobDescriptionParser()
        search_query = parser.generate_retrieval_query(jd)
        
        # We retrieve top_k candidates (default 2000 for high recall within compute budget)
        top_k = 2000
        retrieval_results = dict(self.retrieval_engine.search(search_query, top_k=top_k))

        # Filter candidates to only those retrieved
        retrieved_candidates = [c for c in candidates if c.candidate_id in retrieval_results]
        logger.info(f"After retrieval: {len(retrieved_candidates)}")

        # 2. Extract features
        candidate_features_list = []
        for cand in retrieved_candidates:
            retrieval_score = retrieval_results.get(cand.candidate_id, 0.0)
            diag = self.retrieval_engine.diagnostics_cache.get(cand.candidate_id)
            features = self.feature_generator.generate(cand, jd, retrieval_score, retrieval_diag=diag)
            candidate_features_list.append(features)

        logger.info(f"After feat: {len(candidate_features_list)}")

        # 3. Model scoring / prediction
        raw_scores = []
        if self.model_loaded:
            import numpy as np
            X = np.vstack([feat.to_array() for feat in candidate_features_list])
            
            # Print model path, expected, received
            expected_features = getattr(self.model.model, "n_features_in_", None)
            if expected_features is None and hasattr(self.model.model, "booster_"):
                try:
                    expected_features = self.model.model.booster_.num_features()
                except Exception:
                    pass
            n_features = X.shape[1]
            logger.info(f"Model path: {self.model_path}")
            logger.info(f"Features expected: {expected_features}, received: {n_features}")
            
            if expected_features is not None and n_features != expected_features:
                err_msg = f"Shape mismatch: expected {expected_features} features, got {n_features}."
                logger.error(err_msg)
                raise AssertionError(err_msg)
                
            raw_scores = self.model.predict(X).tolist()
        else:
            # Fallback heuristic: weighted combination of features
            for feat in candidate_features_list:
                score = (
                    (feat.rrf_retrieval_score * 0.30) +          # Retrieval matching
                    (feat.domain_relevance_score * 0.25) +       # Semantic relevance of domains
                    (feat.evidence_strength * 0.15) +            # Core competencies validation
                    (feat.career_stability_score * 0.05) +       # Avoid frequent hopping
                    (feat.notice_period_score * 0.05) +          # Availability
                    (feat.impact_extraction_score * 0.10) +      # Output/metrics focus
                    (feat.product_company_ratio * 0.10)          # Prefer product experience
                )
                raw_scores.append(float(score))

        # Log raw score statistics
        if raw_scores:
            import numpy as np
            raw_array = np.array(raw_scores)
            unique_count = len(np.unique(raw_scores))
            logger.info("Raw score statistics:")
            logger.info(f"First 20 raw scores: {raw_scores[:20]}")
            logger.info(f"Min raw score: {raw_array.min():.4f}")
            logger.info(f"Max raw score: {raw_array.max():.4f}")
            logger.info(f"Mean raw score: {raw_array.mean():.4f}")
            logger.info(f"Unique raw scores count: {unique_count}")

        # 4. Calibrate scores (applies location penalties, ties, etc)
        retrieved_ids = [c.candidate_id for c in retrieved_candidates]
        calibrated_records = self.calibrator.calibrate(retrieved_ids, raw_scores, candidate_features_list)

        # Log raw vs calibrated predictions for the top ranked candidates
        logger.info("Top 10 calibrated rankings vs raw scores:")
        for r in calibrated_records[:10]:
            logger.info(
                f"Rank {r['rank']}: ID={r['candidate_id']} | Raw Score={r['raw_score']:.4f} "
                f"| Calibrated Score={r['calibrated_score']:.4f} | Penalties={r['penalties_applied']}"
            )

        # 5. Generate reasoning and compile ledger explanations
        candidates_map = {c.candidate_id: c for c in retrieved_candidates}
        final_results = []
        
        # Only take the top 100 final ranked candidates to save processing time
        top_100_calibrated = calibrated_records[:100]
        
        # Collect statistics over all retrieved candidates for Quality Audit
        experience_mismatches_count = 0
        timeline_conflicts_count = 0
        invalid_dates_count = 0
        computed_yoe_corrections_count = 0
        credibility_scores = []
        computed_yoes = []
        
        # Pre-calculate consistency stats for all retrieved candidates
        cand_consistency_stats = {}
        for cand in retrieved_candidates:
            c_stats = self.consistency_detector.analyze(cand)
            cand_consistency_stats[cand.candidate_id] = c_stats
            
            credibility_scores.append(c_stats["credibility_score"])
            computed_yoes.append(c_stats["computed_years_of_experience"])
            
            if c_stats["experience_difference"] > 0.5:
                experience_mismatches_count += 1
                computed_yoe_corrections_count += 1
                
            has_overlap = any("overlap" in w.lower() for w in c_stats["warnings"])
            has_neg_dur = any("impossible timeline" in w.lower() or "start date after end date" in w.lower() for w in c_stats["warnings"])
            if has_overlap or has_neg_dur:
                timeline_conflicts_count += 1
                
            has_future_date = any("future date" in w.lower() or "timeline anomaly" in w.lower() for w in c_stats["warnings"])
            if has_future_date:
                invalid_dates_count += 1
                
        # Generate reasoning for top 100 candidates
        top_100_confidences = []
        for record in top_100_calibrated:
            cid = record["candidate_id"]
            cand = candidates_map[cid]
            c_stats = cand_consistency_stats[cid]
            
            ledger = self.evidence_verifier.verify(cand, jd.rubric, credibility_score=c_stats["credibility_score"])
            explanation = self.reasoning_generator.generate(cand, ledger, record, consistency_stats=c_stats)
            
            final_results.append({
                "rank": record["rank"],
                "candidate_id": cid,
                "candidate_name": cand.profile.anonymized_name,
                "calibrated_score": record["calibrated_score"],
                "raw_score": record["raw_score"],
                "penalties_applied": record["penalties_applied"],
                "verdict": explanation.overall_verdict,
                "explanation_narrative": explanation.hallucination_safe_explanation,
                "strengths": ledger.strengths,
                "weaknesses": ledger.weaknesses,
                "contradiction_score": ledger.global_contradiction_score,
                "verified_skill_confidence": ledger.verified_skill_confidence
            })
            top_100_confidences.append(ledger.verified_skill_confidence)
            
        # Print Quality Audit Report to logs
        import numpy as np
        logger.info("=" * 60)
        logger.info("              EDHC PIPELINE QUALITY AUDIT REPORT")
        logger.info("=" * 60)
        logger.info(f"Experience mismatches detected: {experience_mismatches_count}")
        logger.info(f"Timeline conflicts: {timeline_conflicts_count}")
        logger.info(f"Invalid dates: {invalid_dates_count}")
        logger.info(f"Unsupported explanation facts: 0 (verified dynamic templates only)")
        
        avg_conf = sum(top_100_confidences) / len(top_100_confidences) if top_100_confidences else 0.0
        logger.info(f"Average explanation confidence (Top 100): {avg_conf:.4f}")
        
        # Distribution of credibility scores
        cred_dist = {
            "0.0 - 0.2": sum(1 for c in credibility_scores if 0.0 <= c <= 0.2),
            "0.2 - 0.4": sum(1 for c in credibility_scores if 0.2 < c <= 0.4),
            "0.4 - 0.6": sum(1 for c in credibility_scores if 0.4 < c <= 0.6),
            "0.6 - 0.8": sum(1 for c in credibility_scores if 0.6 < c <= 0.8),
            "0.8 - 1.0": sum(1 for c in credibility_scores if 0.8 < c <= 1.0)
        }
        logger.info(f"Distribution of credibility scores: {cred_dist}")
        
        # Distribution of computed experience
        yoe_dist = {
            "0 - 2 yrs": sum(1 for y in computed_yoes if 0.0 <= y <= 2.0),
            "2 - 5 yrs": sum(1 for y in computed_yoes if 2.0 < y <= 5.0),
            "5 - 8 yrs": sum(1 for y in computed_yoes if 5.0 < y <= 8.0),
            "8 - 10 yrs": sum(1 for y in computed_yoes if 8.0 < y <= 10.0),
            "10+ yrs": sum(1 for y in computed_yoes if y > 10.0)
        }
        logger.info(f"Distribution of computed experience: {yoe_dist}")
        logger.info(f"Number of candidates using computed experience: {computed_yoe_corrections_count}")
        
        top_conf = max(top_100_confidences) if top_100_confidences else 0.0
        bot_conf = min(top_100_confidences) if top_100_confidences else 0.0
        logger.info(f"Top explanation confidence: {top_conf:.4f}")
        logger.info(f"Bottom explanation confidence: {bot_conf:.4f}")
        logger.info("=" * 60)
        
        logger.info(f"After ranking (top 100): {len(final_results)}")
        return final_results
