import torch
import pandas as pd
import os
import json
import datetime
from torch.utils.data import DataLoader, random_split
from src.preprocessing import preprocess_data
from src.config import Settings, load_settings
from src.dataset import RecipeDataset
from src.models import AblationType, RecipeNet, HeadType
from src.trainer import Trainer, LossFunc

# Configuration
class Config:
    learning_rate = 1e-3
    weight_decay = 1e-5
    batch_size = 64  
    epochs = 1
    hidden_dim = 128
    loss_fn = LossFunc.HUBER # LossFunc.MSE, LossFunc.MAE, or LossFunc.HUBER
    head_type = HeadType.SHALLOW # HeadType.SHALLOW, HeadType.DEEP, or HeadType.RESIDUAL
    ablation = AblationType.ALL_FEATURES # AblationType.META_ONLY, AblationType.TAG_ONLY, or AblationType.ALL_FEATURES
    return_embeddings = False # Set to True to return and save latent embeddings during evaluation

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
    print(f"Training the {cfg.head_type.value} model on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}:")
    trainer = Trainer(model, train_loader, val_loader, cfg)
    history = trainer.fit(cfg.epochs, cfg.head_type, cfg.ablation, cfg.loss_fn)
    
    # Evaluate on Test Set
    print(f"Test set evaluation for {cfg.head_type.value} model:")
    model_path = os.path.join(s.models_dir, f"best_model_{cfg.head_type.value}.pth")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # Generate a unique experiment ID for saving results and embeddings
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{cfg.head_type.value}_{cfg.ablation.value}_{timestamp}"

    # Evaluate and optionally save embeddings
    test_metrics, embeddings = trainer.evaluate(
        test_loader, 
        cfg.head_type, 
        cfg.ablation, 
        return_embeddings=cfg.return_embeddings
    )
    
    # Save Metrics (JSON)
    history.update(test_metrics)
    results_path = os.path.join(s.results_dir, f"results_{experiment_id}.json")
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=4)

    # Save Embeddings (PT) if enabled
    if embeddings is not None:
        embed_path = os.path.join(s.results_dir, f"embeds_{experiment_id}.pt")
        torch.save(embeddings, embed_path)
        print(f"Embeddings saved to {embed_path}")

    print(f"Results saved to {results_path}")
    
    

if __name__ == "__main__":
    main()