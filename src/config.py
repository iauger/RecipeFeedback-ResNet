# src/config.py

"""
Config module for loading and validating environment variables
"""

from __future__ import annotations

import datetime
from datetime import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import platform
import re
import logging

logger = logging.getLogger(__name__)


# Helper functions for path resolution
def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "requirements.txt").exists(): # crude heuristic for repo root
            return p
    # fallback: current behavior
    return here.parents[1]


def resolve_path(path: str | None, default: str) -> str:
    root = repo_root()

    if not path:
        path = default

    path = path.strip()

    p = Path(path)

    if p.is_absolute():
        return str(p)

    return str((root / p).resolve())

@dataclass
class Settings:
    
    env: str

    # Directory paths
    data_dir: str
    raw_dir: str
    processed_dir: str
    src_dir: str
    models_dir: str
    features_dir: str
    # Data file paths
    # Raw data file paths
    raw_recipes_path: str
    raw_reviews_path: str
    raw_labeled_reviews_path: str
    
    # Processed data file paths
    processed_recipes_path: str
    processed_reviews_path: str
    
    
def validate_settings(s: Settings) -> None:
    require_raw = os.getenv("REQUIRE_RAW_INPUTS", "0").strip() == "1"

    if s.env == "local" and require_raw:
        for p in [s.raw_recipes_path, s.raw_reviews_path, s.raw_labeled_reviews_path]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing required file: {p}")

    # still ensure directories exist
    for d in [s.raw_dir, s.processed_dir, s.models_dir, s.features_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)


def load_settings(*, prefer_latest_run: bool = True) -> Settings:
    load_dotenv(override=False)

    env = os.getenv("ENV", "local").strip().lower()
    
    src_dir = resolve_path(os.getenv("SRC_DIR"), "./src")
    data_dir = resolve_path(os.getenv("DATA_DIR"), "./data")
    raw_dir = resolve_path(os.getenv("RAW_DIR"), "./data/raw")
    processed_dir = resolve_path(os.getenv("PROCESSED_DIR"), "./data/processed")
    features_dir = resolve_path(os.getenv("FEATURES_DIR"), "./data/processed/features")
    models_dir = resolve_path(os.getenv("MODELS_DIR"), "./data/models")

    raw_recipes_path = resolve_path(os.getenv("RAW_RECIPES_PATH"), "./data/raw/gold/modeling_recipe.parquet")
    raw_reviews_path = resolve_path(os.getenv("RAW_REVIEWS_PATH"), "./data/raw/gold/modeling_reviews.parquet")
    raw_labeled_reviews_path = resolve_path(os.getenv("RAW_LABELED_REVIEWS_PATH"), "./data/raw/gold/gold_features_1.4M.parquet")
    processed_reviews_path = resolve_path(os.getenv("PROCESSED_REVIEWS_PATH"), "./data/processed/PROCESSED_reviews.parquet")
    processed_recipes_path = resolve_path(os.getenv("PROCESSED_RECIPES_PATH"), "./data/processed/PROCESSED_recipes.parquet")

    raw_dir = resolve_path(os.getenv("RAW_DIR"), "./data/raw")
    processed_dir = resolve_path(os.getenv("PROCESSED_DIR"), "./data/processed")
    features_dir = resolve_path(os.getenv("FEATURES_DIR"), "./data/processed/features")
    
    s = Settings(
        env=env,
        src_dir=src_dir,
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        features_dir=features_dir,
        models_dir=models_dir,
        raw_recipes_path=raw_recipes_path,
        raw_reviews_path=raw_reviews_path,
        raw_labeled_reviews_path=raw_labeled_reviews_path,
        processed_reviews_path=processed_reviews_path,
        processed_recipes_path=processed_recipes_path
    )
    validate_settings(s)
    return s