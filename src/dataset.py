# src/dataset.py

"""
Dataset module for ensure features are correctly mapped to appropriate tensors
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class RecipeDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """
        Group features into metadata and tags
        """
        # Metadata tensors
        self.meta_cols = [
            'minutes', 'n_steps', 'n_ingredients', 'calories', 'fat', 
            'sugar', 'sodium', 'protein', 'saturated_fat', 'carbs'
        ] + [c for c in df.columns if c.startswith(('cat_', 'ing_'))]
        
        # Tag tensors
        self.tag_cols = [c for c in df.columns if c.startswith(('pred_', 'intensity_'))]
        
        # Label
        self.targets = torch.tensor(df['bayesian_rating'].values, dtype=torch.float32).view(-1, 1)
        
        # Convert features to tensors
        self.meta_features = torch.tensor(df[self.meta_cols].values, dtype=torch.float32)
        self.tag_features = torch.tensor(df[self.tag_cols].values, dtype=torch.float32)
        self.recipe_ids = df['recipe_id'].values
        self.recipe_name = df['name'].values
        
    @property
    def meta_dim(self) -> int:
        """Returns the number of metadata/structural features."""
        return len(self.meta_cols)

    @property
    def tag_dim(self) -> int:
        """Returns the number of NLP/experiential features."""
        return len(self.tag_cols)
        

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
        # Return a tuple of (metadata features, tag features, target, recipe_ids, recipe_name)
        meta_features = self.meta_features[idx]
        tag_features = self.tag_features[idx]
        target = self.targets[idx]
        recipe_id = self.recipe_ids[idx]
        recipe_name = self.recipe_name[idx]
        return meta_features, tag_features, target, recipe_id, recipe_name