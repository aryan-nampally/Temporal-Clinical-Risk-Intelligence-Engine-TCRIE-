"""
TCRIE - Analysis Persistence Layer
Stores and retrieves analysis results as JSON files so that
results survive API restarts.
Author: Nampally Aryan
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache"


def _ensure_cache_dir() -> None:
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def save_analysis(study_id: str, data: Dict[str, Any]) -> None:
    """Persist a study analysis result to disk as JSON."""
    _ensure_cache_dir()
    filepath = CACHE_DIR / f"{study_id}.json"
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Persisted analysis: {study_id} -> {filepath}")


def load_analysis(study_id: str) -> Optional[Dict[str, Any]]:
    """Load a previously persisted analysis result."""
    filepath = CACHE_DIR / f"{study_id}.json"
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Cache read failed for {study_id}: {e}")
        return None


def list_cached_studies() -> List[str]:
    """Return study IDs that have cached results on disk."""
    _ensure_cache_dir()
    return [p.stem for p in CACHE_DIR.glob("*.json")]
