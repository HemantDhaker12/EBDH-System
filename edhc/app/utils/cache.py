import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional
from edhc.app.config.settings import settings
from edhc.app.utils.logger import get_logger

logger = get_logger(__name__)

class FileCache:
    """A file-based key-value cache that supports JSON or Pickle formats."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or settings.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_filepath(self, key: str, use_pickle: bool) -> Path:
        """Hash key and return absolute file path."""
        hashed_key = hashlib.md5(key.encode("utf-8")).hexdigest()
        ext = ".pkl" if use_pickle else ".json"
        return self.cache_dir / f"{hashed_key}{ext}"

    def get(self, key: str, use_pickle: bool = False) -> Optional[Any]:
        """Retrieve a value from the cache."""
        filepath = self._get_filepath(key, use_pickle)
        if not filepath.exists():
            return None
            
        try:
            if use_pickle:
                with open(filepath, "rb") as f:
                    return pickle.load(f)
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read cache file {filepath}: {e}")
            return None

    def set(self, key: str, value: Any, use_pickle: bool = False) -> bool:
        """Store a value in the cache."""
        filepath = self._get_filepath(key, use_pickle)
        try:
            if use_pickle:
                with open(filepath, "wb") as f:
                    pickle.dump(value, f)
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(value, f, default=str)
            return True
        except Exception as e:
            logger.warning(f"Failed to write cache file {filepath}: {e}")
            return False

    def delete(self, key: str, use_pickle: bool = False) -> bool:
        """Remove an item from cache."""
        filepath = self._get_filepath(key, use_pickle)
        if filepath.exists():
            try:
                filepath.unlink()
                return True
            except Exception as e:
                logger.warning(f"Failed to delete cache file {filepath}: {e}")
        return False

    def clear(self) -> None:
        """Clear all cache files."""
        for path in self.cache_dir.glob("*"):
            if path.is_file():
                try:
                    path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete {path}: {e}")
