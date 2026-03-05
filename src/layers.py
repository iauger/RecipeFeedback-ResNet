import torch
import torch.nn as nn

"""
Layers module for defining reusable building blocks of the neural network architecture, such as fully connected blocks and residual blocks.
"""

# Reusable fully connected block with normalization, activation, and dropout
class FullyConnectedBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int, dropout: float = 0.2):
        """
        Standard fully connected block with linear transformation, batch normalization, ReLU activation, and dropout.
        """
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.batchnorm = nn.BatchNorm1d(out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


# Reusable residual block with skip connections
class ResidualBlock(nn.Module):
    def __init__(self, size: int, dropout: float = 0.2):
        """
        Residual block that applies two fully connected layers with a skip connection. 
        The input is added to the output of the two layers before applying a final ReLU activation.
        """
        super().__init__()
        # Two linear layers with normalization in between
        self.path = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(size, size),
            nn.BatchNorm1d(size) 
        )

        # ReLU for the final combined output
        self.final_relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip connection logic
        identity = x
        out = self.path(x)
        out += identity  # Add skip connection
        out = self.final_relu(out)  # Final activation after addition
        return out