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
        self.df = df
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
        

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Return a tuple of (metadata features, tag features, target)
        meta_features = self.meta_features[idx]
        tag_features = self.tag_features[idx]
        target = self.targets[idx]
        return meta_features, tag_features, target