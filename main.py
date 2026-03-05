import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from src.preprocessing import preprocess_data
from src.config import Settings, load_settings
from src.dataset import RecipeDataset
from src.models import RecipeNet
from src.trainer import Trainer

# 1. Configuration - Use a Simple Namespace or Dictionary
class Config:
    learning_rate = 1e-3
    weight_decay = 1e-5
    batch_size = 64  
    epochs = 10
    hidden_dim = 128

def main():
    # Establish configuration and settings
    cfg = Config()
    s = load_settings()
    
    # Process data
    df = preprocess_data(s)
    
    # Setup Dataset and Splits
    full_dataset = RecipeDataset(df)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, num_workers=0)
    
    # Initialize Model
    model = RecipeNet(meta_in=full_dataset.meta_dim, tag_in=full_dataset.tag_dim, hidden_dim=cfg.hidden_dim)
    
    # Train
    print(f"Starting training on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}...")
    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.fit(cfg.epochs)

if __name__ == "__main__":
    main()