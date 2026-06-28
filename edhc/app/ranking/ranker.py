import joblib
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

from edhc.app.config.settings import settings
from edhc.app.utils.logger import get_logger

logger = get_logger(__name__)

class BaseRanker(ABC):
    """Abstract base class for all ranking models in the EDHC ecosystem."""

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, qids: np.ndarray) -> None:
        """Train the ranker.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Relevance labels of shape (n_samples,)
            qids: Query IDs for grouping of shape (n_samples,)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict relevance scores for candidate feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
        Returns:
            Scores array of shape (n_samples,)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialize model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model serialization from disk."""
        pass


class LambdaMARTRanker(BaseRanker):
    """Learning-to-Rank implementation using LightGBM LambdaMART."""

    def __init__(self, params: Optional[dict] = None) -> None:
        self.params = params or {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5],
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "n_estimators": 100,
            "verbose": -1,
            "random_state": 42
        }
        self.model = lgb.LGBMRanker(**self.params)

    def train(self, X: np.ndarray, y: np.ndarray, qids: np.ndarray) -> None:
        """Train LGBMRanker using query groupings."""
        logger.info(f"Training LambdaMART model on {X.shape[0]} candidate samples...")
        
        # Sort data by query ID because LightGBM expects grouped queries to be contiguous
        sort_indices = np.argsort(qids)
        X_sorted = X[sort_indices]
        y_sorted = y[sort_indices]
        qids_sorted = qids[sort_indices]

        # Calculate groups count sizes
        _, group_sizes = np.unique(qids_sorted, return_counts=True)
        
        self.model.fit(
            X=X_sorted,
            y=y_sorted,
            group=group_sizes
        )
        logger.info("LambdaMART model training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict relevance score matrix."""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
        
        # Check fitted features count for backward compatibility slicing
        n_features = X.shape[1]
        expected_features = getattr(self.model, "n_features_in_", None)
        if expected_features is None and hasattr(self.model, "booster_"):
            try:
                expected_features = self.model.booster_.num_features()
            except Exception:
                pass
                
        if expected_features is not None and n_features > expected_features:
            logger.info(f"Slicing input features from {n_features} to LambdaMART expected {expected_features}")
            X = X[:, :expected_features]
            
        return self.model.predict(X)

    def save(self, path: str) -> None:
        """Save model object."""
        logger.info(f"Saving LambdaMART model to {path}")
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        """Load model object."""
        logger.info(f"Loading LambdaMART model from {path}")
        self.model = joblib.load(path)


class LinearRanker(BaseRanker):
    """Fallback simple linear regression baseline ranker."""

    def __init__(self) -> None:
        self.model = LinearRegression()

    def train(self, X: np.ndarray, y: np.ndarray, qids: np.ndarray) -> None:
        logger.info("Training Linear Baseline Ranker...")
        # Baseline regression ignores group IDs
        self.model.fit(X, y)
        logger.info("Linear Ranker training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
            
        n_features = X.shape[1]
        expected_features = getattr(self.model, "n_features_in_", None)
        if expected_features is not None and n_features > expected_features:
            logger.info(f"Slicing input features from {n_features} to Linear expected {expected_features}")
            X = X[:, :expected_features]
            
        return self.model.predict(X)

    def save(self, path: str) -> None:
        logger.info(f"Saving Linear Ranker to {path}")
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        logger.info(f"Loading Linear Ranker from {path}")
        self.model = joblib.load(path)
