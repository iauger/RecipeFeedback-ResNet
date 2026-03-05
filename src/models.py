import torch
import torch.nn as nn

from src.layers import FullyConnectedBlock, ResidualBlock

class RecipeNet(nn.Module):
    def __init__(self, meta_in: int, tag_in: int, hidden_dim: int = 128):
        """
        Dual-Encoder architecture with a Residual Backbone.
        - Metadata Encoder: Processes continuous and one-hot encoded features related to the recipe's metadata (e.g., cooking time, calories, recipe tags, ingredients).
        - Tag Encoder: Processes semantic feedback tags from reviews (e.g., "too_salty", "easy_quick", "family_hit").
        """
        super().__init__()        
        # Initialization of the architecture components
        fc = FullyConnectedBlock
        res = ResidualBlock
        
        # Metadata Encoder
        # Moves features from 209 to 128
        self.meta_encoder = nn.Sequential(
            fc(meta_in, hidden_dim * 2),
            fc(hidden_dim * 2, hidden_dim)
        )
        
        # Tag Encoder
        # Expands from 34 to 128 to match the metadata encoder's output dimension
        self.tag_encoder = fc(tag_in, hidden_dim)   
        
        # Encoder Fusion & Backbone
        # After encoding, we concatenate the outputs (128 + 128 = 256) and pass through a backbone of fully connected and residual blocks
        fusion_dim = hidden_dim * 2
        self.backbone = nn.Sequential(
            fc(fusion_dim, fusion_dim),
            res(fusion_dim),
            res(fusion_dim),
            fc(fusion_dim, hidden_dim)
        )
        
        # Output Head
        # Final regression layer to predict the Bayesian rating
        self.regressor = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.regressor.weight)
        if self.regressor.bias is not None:
            nn.init.zeros_(self.regressor.bias)
        

    def forward(self, meta_x, tag_x):
        meta_out = self.meta_encoder(meta_x)
        tag_out = self.tag_encoder(tag_x)
        fused = torch.cat([meta_out, tag_out], dim=1)
        backbone_out = self.backbone(fused)
        return self.regressor(backbone_out)