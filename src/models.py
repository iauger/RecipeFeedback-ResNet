import torch
import torch.nn as nn

from src.layers import FullyConnectedBlock, ResidualBlock

class RecipeNet(nn.Module):
    def __init__(self, meta_in: int, tag_in: int, hidden_dim: int = 128):
        """
        Dual-Encoder architecture with a Residual Backbone.
        
        Args:
            meta_in: Number of metadata features (209).
            tag_in: Number of tag features (34).
            hidden_dim: The size of the shared latent space.
        """
        super().__init__()
        
        # STUB 1: Metadata Encoder (The Bottleneck)
        # Goal: Transition from sparse 209-dim to a dense hidden_dim.
        # Hint: Use a two-step reduction (e.g., meta_in -> hidden_dim*2 -> hidden_dim).
        self.meta_encoder = None 
        
        # STUB 2: Tag Encoder (The Refiner)
        # Goal: Project 34-dim NLP signals into a dense hidden_dim.
        # Hint: Since this is already dense, a single block may suffice.
        self.tag_encoder = None
        
        # STUB 3: Feature Fusion & Backbone
        # Goal: Concatenate the two encoders and reason through them.
        # Hint: input size here will be hidden_dim * 2.
        self.backbone = None
        
        # STUB 4: The Output Head
        # Goal: Final regression to the Bayesian Rating.
        self.regressor = None

    def forward(self, meta_x, tag_x):
        # STUB 5: The Forward Pass logic
        pass