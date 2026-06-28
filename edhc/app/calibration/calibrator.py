from typing import List, Dict, Any
from edhc.app.schemas.features import CandidateFeatures
from edhc.app.utils.logger import get_logger

logger = get_logger(__name__)

class ScoreCalibrator:
    """Normalizes candidate ranking scores, applies business logic penalties, and resolves ranking ties deterministically."""

    def calibrate(
        self,
        candidate_ids: List[str],
        raw_scores: List[float],
        features_list: List[CandidateFeatures]
    ) -> List[Dict[str, Any]]:
        """Normalize raw predictions, apply penalties, and return sorted scores list.

        Returns:
            List of dicts containing:
            [
                {
                    "candidate_id": "xyz",
                    "raw_score": 0.85,
                    "calibrated_score": 0.72,
                    "penalties_applied": ["contradiction_penalty"],
                    "rank": 1
                }
            ]
        """
        logger.info(f"Calibrating scores for {len(candidate_ids)} candidates...")
        
        if not candidate_ids:
            return []

        # 1. Normalize raw scores using min-max scaling to [0.0, 1.0]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        score_range = max_score - min_score
        
        normalized_scores = []
        for s in raw_scores:
            if score_range > 0:
                normalized_scores.append((s - min_score) / score_range)
            else:
                normalized_scores.append(0.5)  # Default average score if all are identical

        # Create mapping of candidate ID to features
        features_map = {f.candidate_id: f for f in features_list}

        # 2. Apply Penalties
        calibrated_records = []
        for idx, cid in enumerate(candidate_ids):
            base_score = normalized_scores[idx]
            features = features_map.get(cid)
            
            penalties = []
            final_score = base_score
            
            if features:
                # Severe Honeypot penalty
                if features.credibility_score < 0.05:
                    final_score -= 2.0
                    penalties.append("honeypot_contradiction_penalty (-2.0)")
                # Low credibility warning penalty
                elif features.credibility_score <= 0.7 or getattr(features, "contradiction_count", 0) > 0:
                    final_score -= 0.15
                    penalties.append("low_credibility_penalty (-0.15)")
                    
                # Pure consulting/services experience penalty
                if features.services_company_ratio >= 0.99:
                    final_score -= 0.25
                    penalties.append("services_only_company_penalty (-0.25)")
                elif features.services_company_ratio > 0.5:
                    final_score -= 0.10
                    penalties.append("high_services_company_penalty (-0.10)")
                    
                # Long notice period penalty
                if features.notice_period_score <= 0.2:
                    final_score -= 0.10
                    penalties.append("long_notice_period_penalty (-0.10)")

            # Clamp calibrated score to [0.0, 1.0]
            final_score = max(0.0, min(1.0, final_score))
            
            calibrated_records.append({
                "candidate_id": cid,
                "raw_score": round(raw_scores[idx], 4),
                "calibrated_score": round(final_score, 4),
                "penalties_applied": penalties,
                "features": features
            })

        # 3. Resolve Ties and Sort
        # Sort by calibrated_score descending (using negative value), and then candidate_id ascending
        calibrated_records.sort(key=lambda x: (-x["calibrated_score"], x["candidate_id"]))

        # Apply a tiny strictly monotonic decreasing tie-breaker to ensure no identical scores in the final 4-decimal formatted output
        for i, item in enumerate(calibrated_records):
            val = item["calibrated_score"]
            if i < 100:
                val = max(0.01, val)
            item["calibrated_score"] = max(0.0, round(val - (i * 0.0001), 4))

        # Assign ranks
        for rank, item in enumerate(calibrated_records, 1):
            item["rank"] = rank
            # Remove features dictionary to keep output clean
            if "features" in item:
                del item["features"]

        logger.info("Score calibration complete.")
        return calibrated_records

