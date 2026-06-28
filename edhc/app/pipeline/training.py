import numpy as np
from typing import List, Dict, Any, Optional

from edhc.app.config.settings import settings
from edhc.app.schemas.candidate import CandidateProfile
from edhc.app.schemas.jd import JobDescriptionParsed
from edhc.app.features.generator import FeatureGenerator
from edhc.app.ranking.ranker import LambdaMARTRanker
from edhc.app.utils.logger import get_logger

logger = get_logger(__name__)

class TrainingPipeline:
    """Coordinates training of the LTR model by extracting features and fitting LightGBM Rankers."""

    def __init__(self) -> None:
        self.feature_generator = FeatureGenerator()
        self.ranker = LambdaMARTRanker()

    def run(self, train_samples: List[Dict[str, Any]], model_save_path: Optional[str] = None) -> None:
        """Run training pipeline.

        train_samples is expected to be a list of dictionaries:
        [
            {
                "candidate": CandidateProfile,
                "jd": JobDescriptionParsed,
                "label": int,       # Relevance score (e.g., 0 to 4)
                "query_id": int,    # Query grouping ID for LambdaMART
                "retrieval_score": float  # Optional base retrieval score
            }
        ]
        """
        logger.info(f"Starting LTR model training with {len(train_samples)} samples...")
        
        if not train_samples:
            logger.warning("No training samples provided. Skipping training.")
            return

        features_list = []
        labels = []
        qids = []

        for sample in train_samples:
            candidate: CandidateProfile = sample["candidate"]
            jd: JobDescriptionParsed = sample["jd"]
            label: int = sample["label"]
            qid: int = sample["query_id"]
            retrieval_score: float = sample.get("retrieval_score", 0.5)
            retrieval_diag: Optional[dict] = sample.get("retrieval_diag", None)

            # Generate structured features
            features = self.feature_generator.generate(
                candidate, jd, retrieval_score, retrieval_diag=retrieval_diag
            )
            
            features_list.append(features.to_array())
            labels.append(label)
            qids.append(qid)

        # Convert to numpy arrays
        X = np.vstack(features_list)
        y = np.array(labels, dtype=np.int32)
        qids_array = np.array(qids, dtype=np.int32)

        # Fit Ranker
        self.ranker.train(X, y, qids_array)

        # Save model
        resolved_save_path = model_save_path or str(settings.MODELS_DIR / "lambdamart_model.pkl")
        
        # Ensure directories exist
        import os
        os.makedirs(os.path.dirname(resolved_save_path), exist_ok=True)
        
        self.ranker.save(resolved_save_path)
        logger.info(f"LTR ranker model successfully trained and serialized to {resolved_save_path}")

