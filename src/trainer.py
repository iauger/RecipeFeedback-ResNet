# src/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        
        """
        Trainer class to encapsulate the training loop, validation, and performance tracking for the RecipeNet model.
        - model: The RecipeNet model instance to be trained.
        - train_loader: DataLoader for the training dataset.
        - val_loader: DataLoader for the validation dataset.
        - config: Configuration object containing training hyperparameters.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.HuberLoss() 
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.history = {'train_loss': [], 'val_loss': []}
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min')

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for meta_x, tag_x, targets in self.train_loader:
            # Move data to device
            meta_x, tag_x, targets = meta_x.to(self.device), tag_x.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(meta_x, tag_x)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Compute average training loss and store in history
        avg_loss = total_loss / len(self.train_loader)
        self.history['train_loss'].append(avg_loss)
        return avg_loss            
                
    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for meta_x, tag_x, targets in self.val_loader:
                # Move data to device
                meta_x, tag_x, targets = meta_x.to(self.device), tag_x.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(meta_x, tag_x)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        # Compute average validation loss and store in history
        avg_loss = total_loss / len(self.val_loader)
        self.history['val_loss'].append(avg_loss)
        return avg_loss

    def fit(self, epochs: int):
    
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Checkpoint
            # Save the model if validation loss has improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                status = "(Improved! Saved)"
            else:
                status = ""

            print(f"Epoch {epoch:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} {status}")
            
            self.scheduler.step(val_loss)