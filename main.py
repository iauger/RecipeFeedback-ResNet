import torch
import pandas as pd
import os
import json
import datetime
from torch.utils.data import DataLoader, random_split
from src.preprocessing import preprocess_data
from src.config import Settings, load_settings
from src.dataset import RecipeDataset
from src.models import RecipeNet, HeadType
from src.trainer import Trainer

# Configuration
class Config:
    learning_rate = 1e-3
    weight_decay = 1e-5
    batch_size = 64  
    epochs = 5
    hidden_dim = 128
    head_type = HeadType.SHALLOW

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
    model = RecipeNet(
        meta_in=full_dataset.meta_dim, 
        tag_in=full_dataset.tag_dim, 
        hidden_dim=cfg.hidden_dim, 
        head_type=cfg.head_type
        )
    
    # Train
    print(f"Starting training the {cfg.head_type.value} model on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}...")
    trainer = Trainer(model, train_loader, val_loader, cfg)
    history = trainer.fit(cfg.epochs, cfg.head_type)
    
    # Evaluate on Test Set
    print(f"Test set evaluation for {cfg.head_type.value} model:")
    model_path = os.path.join(s.models_dir, f"best_model_{cfg.head_type.value}.pth")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    test_metrics = trainer.evaluate(test_loader, cfg.head_type)
    
    history.update(test_metrics)
    
    results_path = os.path.join(s.results_dir, f"results_{cfg.head_type.value}_{datetime.datetime.now().timestamp()}.json")
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"Experiment results saved to {results_path}")

if __name__ == "__main__":
    main()