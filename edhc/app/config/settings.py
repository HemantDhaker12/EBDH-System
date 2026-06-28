import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EDHC_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )

    # Base Paths
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data")
    RAW_DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "raw")
    PROCESSED_DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "processed")
    EMBEDDINGS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "embeddings")
    CACHE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "caches")
    MODELS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "models")
    OUTPUT_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "outputs")

    # Embedding and Search Config
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    HYBRID_RETRIEVAL_ALPHA: float = 0.5  # Weight for dense embeddings in hybrid scoring

    # Feature Engineering Config
    MAX_CANDIDATE_EXPERIENCE_YEARS: float = 40.0
    MIN_STABILITY_THRESHOLD_YEARS: float = 1.5

    # API Config
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    def create_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        for path in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.EMBEDDINGS_DIR,
            self.CACHE_DIR,
            self.MODELS_DIR,
            self.OUTPUT_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)

# Instantiate settings
settings = Settings()
