# src/preprocessing.py

"""
Preprocessing module for ingestion, transformation and scaling of the recipe and review data.
    - load_data: Load the recipe and review data from CSV files.
    - recipe_aggregation: Aggregate the recipe data to create a single row per recipe with relevant features.
    - review_aggregation: Aggregate the review data to create a single row per recipe with compressed and aggregated review features.
    - merge_data: Merge the aggregated recipe and review data into a single DataFrame.
    - scale_features: Scale numerical features using StandardScaler or MinMaxScaler.
    - bayesian_rating: Calculate the Bayesian average rating for each recipe to account for varying numbers of reviews and ratings.
    - preprocess_data: Orchestration function for the entire preprocessing pipeline.
    - preprocess_report: Generate a report on the preprocessing steps, including data quality and feature distributions.
"""

import os
import json
import numpy as np
import pandas as pd 
from typing import Tuple, cast
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MultiLabelBinarizer

from src.config import Settings, load_settings

def load_data(s: Settings) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the recipe and review data from Parquet files and return as DataFrames
    """
    recipes_df = pd.read_parquet(s.raw_recipes_path)
    reviews_df = pd.read_parquet(s.raw_reviews_path)
    labels_df = pd.read_parquet(s.raw_labeled_reviews_path)
    
    if recipes_df.empty:
        raise ValueError("Recipes DataFrame is empty.")
    if reviews_df.empty:
        raise ValueError("Reviews DataFrame is empty.")
    if labels_df.empty:
        raise ValueError("Labeled Reviews DataFrame is empty.")
    
    return recipes_df, reviews_df, labels_df

def bayesian_rating(df, global_avg_rating, rating_col='rating', review_count_col='review_count', m_threshold=None):
    """
    Compute target variable using Bayesian average rating
    """
    C = global_avg_rating  
    if m_threshold is None:
        m_threshold = df[review_count_col].quantile(0.25) 
    
    v = df[review_count_col]
    R = df[rating_col]
    
    b_rating = (v * R + m_threshold * C) / (v + m_threshold)
    return b_rating\

def review_aggregation(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates reviews into recipe-level features preserving signal intensity.
    - Perc: Proportion of reviews where pred_tag == 1
    - Intensity: Average sim_tag for reviews where pred_tag == 1
    """
    # Force recipe_id to be string for consistent merging later
    reviews_df['recipe_id'] = reviews_df['recipe_id'].astype(str)
    
    # Calculate global average rating for Bayesian calculation later
    global_avg_rating = reviews_df['rating'].mean()
    
    # Identify the base tag names
    tags = [col.replace('pred_', '') for col in reviews_df.columns if col.startswith('pred_')]
    
    # Frequency and basic stats logic
    agg_dict = {
        'rating': ['mean', 'count'],
        **{f'pred_{tag}': 'mean' for tag in tags}
    }
    
    # Grouping results in recipe_id becoming the index
    recipe_level_df = reviews_df.groupby('recipe_id').agg(agg_dict)
    
    # Calculate Intensity (Similarity) preserving signal intensity
    for tag in tags:
        # Average similarity only where the tag was actually predicted
        intensity = (
            reviews_df[reviews_df[f'pred_{tag}'] == 1]
            .groupby('recipe_id')[f'sim_{tag}']
            .mean()
        )
        recipe_level_df[f'intensity_{tag}'] = intensity
    
    recipe_level_df = recipe_level_df.fillna(0)
    
    # Flatten Multi-index columns
    recipe_level_df.columns = [
        f"{c[0]}_{c[1]}" if isinstance(c, tuple) and c[1] != 'mean' else c[0] 
        for c in recipe_level_df.columns
    ]
    
    # Rename for clarity and reset index to make recipe_id a column again
    recipe_level_df = recipe_level_df.rename(columns={
        'rating': 'raw_mean_rating',
        'rating_count': 'review_count'
    }).reset_index()
    
    # Calculate Bayesian rating
    recipe_level_df['bayesian_rating'] = bayesian_rating(recipe_level_df, global_avg_rating=global_avg_rating, rating_col='raw_mean_rating', review_count_col='review_count', m_threshold=None)
    
    # Define lists to match the flattened names
    all_cols = recipe_level_df.columns.tolist()
    
    # Group existing columns by pattern
    p_cols = [c for c in all_cols if c.startswith('pred_')]
    i_cols = [c for c in all_cols if c.startswith('intensity_')]
    base_cols = ['recipe_id', 'raw_mean_rating', 'review_count', 'bayesian_rating']
    
    # Concatenate only what actually exists to avoid KeyError
    final_column_order = base_cols + sorted(p_cols) + sorted(i_cols)
    recipe_level_df = recipe_level_df[final_column_order]
    
    # Filter to keep only recipes with at least one positive signal in pred_tags
    pred_cols = [c for c in recipe_level_df.columns if c.startswith('pred_')]
    has_signals = recipe_level_df[pred_cols].sum(axis=1) > 0
    recipe_level_df = recipe_level_df[has_signals].copy()
    
    return recipe_level_df

def merge_data(recipe_df: pd.DataFrame, review_agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes ID formatting to ensure a successful merge.
    """
    # Defensive casting: convert to float first, then int, then string 
    # This handles "123", 123, and "123.0" consistently
    for df in [recipe_df, review_agg_df]:
        df['recipe_id'] = (
            pd.to_numeric(df['recipe_id'], errors='coerce')
            .fillna(0)
            .astype(int)
            .astype(str)
        )
    
    return pd.merge(recipe_df, review_agg_df, on='recipe_id', how='inner')

def validate_merge(recipe_df: pd.DataFrame, review_agg_df: pd.DataFrame) -> None:
    """
    Diagnostic QA check to validate the merge between recipe metadata and aggregated reviews.
    """
    recipe_ids = set(recipe_df['recipe_id'].unique())
    review_ids = set(review_agg_df['recipe_id'].unique())
    
    intersection = recipe_ids.intersection(review_ids)
    
    print("\n--- Merge QA Report ---")
    print(f"Unique IDs in Recipe Metadata: {len(recipe_ids)}")
    print(f"Unique IDs in Aggregated Reviews: {len(review_ids)}")
    print(f"Intersection Count: {len(intersection)}")
    
    if len(intersection) == 0:
        print("\nERROR: Zero overlap detected. Checking samples...")
        sample_rec = list(recipe_ids)[0] if recipe_ids else "N/A"
        sample_rev = list(review_ids)[0] if review_ids else "N/A"
        print(f"Sample Recipe ID: '{sample_rec}' (Type: {type(sample_rec)})")
        print(f"Sample Review ID: '{sample_rev}' (Type: {type(sample_rev)})")
        
        # Check for the common '.0' float-to-string ghost
        if ".0" in str(sample_rec) or ".0" in str(sample_rev):
            print("ADVICE: Floating point artifact detected. Use .split('.')[0]")
            
    if len(intersection) > 0 and len(intersection) < 100:
        print("WARNING: Very low overlap. Ensure you are using the correct 'Gold' parquet file.")

def scale_features(df: pd.DataFrame, standard_cols: list | None, minmax_cols: list | None) -> pd.DataFrame:
    """
    Scale numerical features using StandardScaler or MinMaxScaler.
    """
    if standard_cols is not None:
        scaler = StandardScaler()
        df[standard_cols] = scaler.fit_transform(df[standard_cols])
    
    if minmax_cols is not None:
        scaler = MinMaxScaler()
        df[minmax_cols] = scaler.fit_transform(df[minmax_cols])
    else:
        raise ValueError("No columns specified for scaling. Please provide at least one of standard_cols or minmax_cols.")
    
    return df

def encode_multi_label_features(
    df: pd.DataFrame, 
    column: str, 
    prefix: str, 
    top_n: int = 100
) -> pd.DataFrame:
    """
    Abstracted one-hot encoding for multi-label text columns.
    """
    # 1. Convert space-separated strings to lists
    item_lists = df[column].str.split()
    
    # 2. Identify the top N most frequent items to manage dimensionality
    all_items = [item for sublist in item_lists for item in sublist]
    top_items = pd.Series(all_items).value_counts().head(top_n).index.tolist()
    
    # 3. Filter lists to only include the top N items
    filtered_items = item_lists.apply(lambda x: [i for i in x if i in top_items])
    
    # 4. Use MultiLabelBinarizer for one-hot encoding
    mlb = MultiLabelBinarizer(classes=top_items, sparse_output=False)
    encoded_data = cast(np.ndarray, mlb.fit_transform(filtered_items))
    
    # 5. Create DataFrame with specific prefix to distinguish feature types
    # Replaces dashes/spaces with underscores to prevent naming issues in model layers
    encoded_df = pd.DataFrame(
        encoded_data.astype(np.int32),
        columns=[f"{prefix}_{item.replace('-', '_').replace(' ', '_')}" for item in top_items],
        index=df.index
    )
    
    return pd.concat([df, encoded_df], axis=1)

def format_for_search(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Elasticsearch-specific array parsing and boolean casting to the raw dataframe."""
    search_df = df.copy()
    
    # Cast predictions to boolean
    pred_cols = [c for c in search_df.columns if c.startswith("pred_")]
    search_df[pred_cols] = search_df[pred_cols].astype(bool)
    
    # Convert ingredients to arrays & remove underscores
    search_df['ingredients_clean'] = search_df['ingredients_clean'].fillna("").apply(
        lambda x: [ing.replace('_', ' ') for ing in str(x).split()]
    )
    
    # Convert tags to simple arrays
    search_df['tags_clean'] = search_df['tags_clean'].fillna("").apply(
        lambda x: str(x).split()
    )
    
    return search_df

def export_static_mapping(df: pd.DataFrame, settings: Settings) -> None:
    """
    Saves a static JSON mapping of column names to indices.
    """
    # Filter for the exact features used by RecipeNet
    model_features = [col for col in df.columns if col not in 
                     ['recipe_id', 'name', 'bayesian_rating', 'raw_mean_rating', 'review_count']]
    
    mapping = {col: i for i, col in enumerate(model_features)}
    
    mapping_path = os.path.join(settings.models_dir, "column_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=4)
    print(f"Static mapping reference saved to {mapping_path}")

def write_preprocessed_data(df: pd.DataFrame, settings: Settings) -> None:
    """
    Write the preprocessed DataFrame to a Parquet file for downstream use.
    """
    df.to_parquet(settings.processed_recipes_path, index=False)
    print(f"Preprocessed data written to {settings.processed_recipes_path}")

def preprocess_data(settings: Settings, overwrite_processed: bool = False) -> pd.DataFrame:
    
    # Write preprocessed data to Parquet
    if overwrite_processed or not os.path.exists(settings.processed_recipes_path):
            
        # Load data
        recipe_df, review_df, label_df = load_data(settings)
        review_agg_df = review_aggregation(label_df)
        validate_merge(recipe_df, review_agg_df)
        merged_df = merge_data(recipe_df, review_agg_df)
        
        # Format for search (Elasticsearch-specific parsing and boolean casting with no normalization or feature engineering) 
        # Will not be used directly in modeling but ensures the raw data is in a consistent format for any search-based applications
        search_df = format_for_search(merged_df)
        search_df.to_parquet(settings.processed_search_path, index=False)
        print(f"Search index data written to {settings.processed_search_path}")
        
        # Scale features
        standard_cols = [
            col for col in merged_df.columns 
            if col not in ['recipe_id', 'raw_mean_rating', 'review_count', 'bayesian_rating', 'name'] 
            and not col.startswith('pred_') 
            and not col.startswith('intensity_')
            and merged_df[col].dtype in ['float64', 'int64'] 
        ]
        minmax_cols = [col for col in merged_df.columns if col.startswith('intensity_')]
        scaled_df = scale_features(merged_df, standard_cols, minmax_cols)
        
        # Encode recipe tags
        encoded_df = encode_multi_label_features(scaled_df, 'tags_clean', 'cat', top_n=100)
        encoded_df = encode_multi_label_features(encoded_df, 'ingredients_clean', 'ing', top_n=100)
        
        # Export static mapping of column names to indices for consistent reference in model input layers
        export_static_mapping(encoded_df, settings)
        
        # Finalize and write preprocessed data
        write_preprocessed_data(encoded_df, settings)
    
    else:
        print(f"File already exists: {settings.processed_recipes_path}")
        encoded_df = pd.read_parquet(settings.processed_recipes_path)

    return encoded_df

def preprocess_report(encoded_df: pd.DataFrame) -> None:
    """
    Generate a report on the preprocessing steps, excluding text columns from statistics.
    """
    print("Preprocessing Report")
    print("====================")
    print(f"Total recipes after merging: {encoded_df['recipe_id'].nunique()}")
    
    # Identify numeric columns for statistics
    numeric_cols = encoded_df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Total numeric features: {len(numeric_cols)}")
    
    print("\nFeature Distributions:")
    for col in numeric_cols:            
        print(f"{col}: "
              f"mean={encoded_df[col].mean():.4f}, "
              f"std={encoded_df[col].std():.4f}, "
              f"min={encoded_df[col].min():.4f}, "
              f"max={encoded_df[col].max():.4f}")

if __name__ == "__main__":
    s = load_settings()
    df = preprocess_data(s, overwrite_processed=True)
    preprocess_report(df)