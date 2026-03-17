# src/config.py

"""
Configuration utilities for resolving project paths and loading runtime settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
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
    best_model_dir: str
    features_dir: str
    results_dir: str
    
    # Data file paths
    # Raw data file paths
    raw_recipes_path: str
    raw_reviews_path: str
    raw_labeled_reviews_path: str
    
    # Processed data file paths
    processed_recipes_path: str
    processed_reviews_path: str
    processed_search_path: str
    
    # Final outputs
    best_model_path: str
    best_model_embeddings_path: str
    best_model_umap_path: str
    
def validate_settings(s: Settings) -> None:
    require_raw = os.getenv("REQUIRE_RAW_INPUTS", "0").strip() == "1"

    if s.env == "local" and require_raw:
        for p in [s.raw_recipes_path, s.raw_reviews_path, s.raw_labeled_reviews_path]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing required file: {p}")

    # still ensure directories exist
    for d in [s.raw_dir, s.processed_dir, s.models_dir, s.features_dir, s.results_dir, s.best_model_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    load_dotenv(override=False)

    env = os.getenv("ENV", "local").strip().lower()
    
    src_dir = resolve_path(os.getenv("SRC_DIR"), "./src")
    data_dir = resolve_path(os.getenv("DATA_DIR"), "./data")
    raw_dir = resolve_path(os.getenv("RAW_DIR"), "./data/raw")
    processed_dir = resolve_path(os.getenv("PROCESSED_DIR"), "./data/processed")
    features_dir = resolve_path(os.getenv("FEATURES_DIR"), "./data/processed/features")
    models_dir = resolve_path(os.getenv("MODELS_DIR"), "./data/models")
    results_dir = resolve_path(os.getenv("RESULTS_DIR"), "./data/models/results")

    raw_recipes_path = resolve_path(os.getenv("RAW_RECIPES_PATH"), "./data/raw/gold/modeling_recipe.parquet")
    raw_reviews_path = resolve_path(os.getenv("RAW_REVIEWS_PATH"), "./data/raw/gold/modeling_reviews.parquet")
    raw_labeled_reviews_path = resolve_path(os.getenv("RAW_LABELED_REVIEWS_PATH"), "./data/raw/gold/gold_labeled_reviews_20260310_135905.parquet")
    processed_reviews_path = resolve_path(os.getenv("PROCESSED_REVIEWS_PATH"), "./data/processed/PROCESSED_reviews.parquet")
    processed_recipes_path = resolve_path(os.getenv("PROCESSED_RECIPES_PATH"), "./data/processed/PROCESSED_recipes.parquet")
    processed_search_path = resolve_path(os.getenv("PROCESSED_SEARCH_PATH"), "./data/processed/PROCESSED_search_recipes.parquet")
    
    best_model_dir = resolve_path(os.getenv("BEST_MODEL_DIR"), "./data/models/best")
    best_model_path = resolve_path(os.getenv("BEST_MODEL_PATH"), f"{best_model_dir}/runs/best_model_residual_v2_all_features_mse.pth")
    best_model_embeddings_path = resolve_path(os.getenv("BEST_MODEL_EMBEDDINGS_PATH"), f"{best_model_dir}/final_residual_v2_embeddings.pt")
    best_model_umap_path = resolve_path(os.getenv("BEST_MODEL_UMAP_PATH"), f"{best_model_dir}/final_residual_v2_umap_projection.npy")
    
    s = Settings(
        env=env,
        src_dir=src_dir,
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        features_dir=features_dir,
        models_dir=models_dir,
        results_dir=results_dir,
        best_model_dir=best_model_dir,
        best_model_path=best_model_path,
        best_model_embeddings_path=best_model_embeddings_path,
        best_model_umap_path=best_model_umap_path,
        raw_recipes_path=raw_recipes_path,
        raw_reviews_path=raw_reviews_path,
        raw_labeled_reviews_path=raw_labeled_reviews_path,
        processed_reviews_path=processed_reviews_path,
        processed_recipes_path=processed_recipes_path,
        processed_search_path=processed_search_path
    )
    validate_settings(s)
    return s