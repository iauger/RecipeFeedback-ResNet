# src/trainer.py

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import Settings, load_settings
from src.models import HeadType



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
        self.history = {
            'model_type': None,
            'train_loss': [], 
            'val_loss': [],
            'grad_norm': []
            }
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min')
    
    def compute_grad_norm(self) -> float:
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        epoch_grad_norms = []
        
        for meta_x, tag_x, targets in self.train_loader:
            # Move data to device
            meta_x, tag_x, targets = meta_x.to(self.device), tag_x.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(meta_x, tag_x)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Capture gradient norm for monitoring
            grad_norm = self.compute_grad_norm()
            epoch_grad_norms.append(grad_norm)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        # Compute average training loss and store in history
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0.0
        self.history['grad_norm'].append(avg_grad_norm)
        
        avg_loss = total_loss / len(self.train_loader)
        self.history['train_loss'].append(avg_loss)
        return avg_loss            
                
    def validate(self) -> float:
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

    def fit(self, epochs: int, head_type: HeadType) -> dict[str, list[float]]:
        s = load_settings()
        self.history['model_type'] = head_type.value
        model_name = os.path.join(s.models_dir, f"best_model_{head_type.value}.pth")
        best_val_loss = float('inf')
        epoch_pbar  = tqdm(range(1, epochs + 1), desc="Training Progress", unit="epoch")
        
        for epoch in epoch_pbar:
            start_time = time.time()
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            epoch_time = time.time() - start_time
            
            # Checkpoint
            # Save the model if validation loss has improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_name)
                status = "(Improved! Saved)"
            else:
                status = ""
            
            epoch_pbar.set_postfix({
                'T-Loss': f"{train_loss:.4f}",
                'V-Loss': f"{val_loss:.4f}",
                'Sec/Epoch': f"{epoch_time:.1f}"
            })

            print(f"\nEpoch {epoch:02d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}{status}")
            
            self.scheduler.step(val_loss)
        
        return self.history
    
    def evaluate(self, loader, head_type: HeadType) -> dict:
        """
        Final evaluation on the unseen test set to report final model performance.
        """
        self.model.eval()
        total_loss = 0
        total_mae = 0
        
        print(f"Evaluating {head_type.value} model on {len(loader.dataset)} test samples:")
        
        with torch.no_grad():
            for meta_x, tag_x, targets in loader:
                meta_x, tag_x, targets = meta_x.to(self.device), tag_x.to(self.device), targets.to(self.device)
                
                outputs = self.model(meta_x, tag_x)
                
                # Use standard MSE for reporting, even if we trained with Huber
                loss = torch.nn.functional.mse_loss(outputs, targets)
                mae = torch.nn.functional.l1_loss(outputs, targets)
                
                total_loss += loss.item()
                total_mae += mae.item()

        avg_mse = total_loss / len(loader)
        avg_rmse = avg_mse ** 0.5
        avg_mae = total_mae / len(loader)
        
        metrics = {
            'test_mse': avg_mse,
            'test_rmse': avg_rmse,
            'test_mae': avg_mae
        }
        
        print("-" * 30)
        print(f"Final Test Metrics for {head_type.value}:")
        print(f"MSE:  {avg_mse:.4f}")
        print(f"RMSE: {avg_rmse:.4f}")
        print(f"MAE:  {avg_mae:.4f}")
        print("-" * 30)
        
        return metrics