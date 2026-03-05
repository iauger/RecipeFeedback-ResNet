import torch
import torch.nn as nn
from enum import Enum
from src.layers import FullyConnectedBlock, ResidualBlock

class HeadType(Enum):
    SHALLOW = "shallow"
    DEEP = "deep"
    RESIDUAL = "residual"

class RecipeNet(nn.Module):
    def __init__(self, meta_in: int, tag_in: int, hidden_dim: int = 128, head_type: HeadType = HeadType.RESIDUAL):
        """
        Dual-Encoder architecture.
        - Metadata Encoder: Processes continuous and one-hot encoded features related to the recipe's metadata (e.g., cooking time, calories, recipe tags, ingredients).
        - Tag Encoder: Processes semantic feedback tags from reviews (e.g., "too_salty", "easy_quick", "family_hit").
        Multi-Head design:
        - Shallow Head: A simple fully connected layer for quick experimentation.
        - Deep Head: A deeper stack of fully connected layers for capturing more complex interactions.
        - Residual Head: Incorporates residual connections to facilitate training deeper architectures without vanishing gradients.
        The final output is a single regression value representing the predicted Bayesian rating of the recipe.
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
        
        # Head selection
        if head_type == HeadType.SHALLOW:
            self.backbone = self.build_shallow_head(fusion_dim, hidden_dim)
        elif head_type == HeadType.DEEP:
            self.backbone = self.build_deep_head(fusion_dim, hidden_dim)
        elif head_type == HeadType.RESIDUAL:
            self.backbone = self.build_residual_head(fusion_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported head type: {head_type}")
        
        # Output Head
        # Final regression layer to predict the Bayesian rating
        self.regressor = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.regressor.weight)
        if self.regressor.bias is not None:
            nn.init.zeros_(self.regressor.bias)
            
    
    def build_shallow_head(self, fusion_dim: int, hidden_dim: int) -> nn.Sequential:
        # Basic single-layer head for quick experimentation and baseline comparisons
        return nn.Sequential(
            FullyConnectedBlock(fusion_dim, hidden_dim),
        )
    
    def build_deep_head(self, fusion_dim: int, hidden_dim: int) -> nn.Sequential:
        layers = []
        # Project to a large processing space
        layers.append(FullyConnectedBlock(fusion_dim, fusion_dim))
        
        # Add 8 intermediate layers to test stability
        for _ in range(8):
            layers.append(FullyConnectedBlock(fusion_dim, fusion_dim))
            
        # Final projection to hidden_dim
        layers.append(FullyConnectedBlock(fusion_dim, hidden_dim))
        return nn.Sequential(*layers)
    
    def build_residual_head(self, fusion_dim: int, hidden_dim: int) -> nn.Sequential:
        # Residual architecture with skip connections to facilitate training deeper networks without vanishing gradients
        return nn.Sequential(
            FullyConnectedBlock(fusion_dim, fusion_dim),
            ResidualBlock(fusion_dim),
            ResidualBlock(fusion_dim),
            FullyConnectedBlock(fusion_dim, hidden_dim)
        )

    def forward(self, meta_x, tag_x):
        meta_out = self.meta_encoder(meta_x)
        tag_out = self.tag_encoder(tag_x)
        fused = torch.cat([meta_out, tag_out], dim=1)
        backbone_out = self.backbone(fused)
        return self.regressor(backbone_out)