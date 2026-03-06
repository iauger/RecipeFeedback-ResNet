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
    learning_rate = 5e-4
    weight_decay = 1e-5
    batch_size = 256  
    epochs = 300
    loss_fn = LossFunc.HUBER # LossFunc.MSE, LossFunc.LOG_CASH, or LossFunc.HUBER
    lr_mult = 10.0 # Multiplier for head learning rate in the optimizer
    hidden_dim = 128
    head_type = HeadType.SHALLOW # HeadType.SHALLOW, HeadType.DEEP, or HeadType.RESIDUAL
    ablation = AblationType.ALL_FEATURES # AblationType.META_ONLY, AblationType.TAG_ONLY, or AblationType.ALL_FEATURES
    return_embeddings = True # Set to True to return and save latent embeddings during evaluation

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
    
    # The Experimental Matrix
    # heads = [HeadType.SHALLOW, HeadType.DEEP, HeadType.RESIDUAL, HeadType.RESIDUAL_V2]
    heads = [HeadType.RESIDUAL_V2]
    ablations = [AblationType.ALL_FEATURES, AblationType.META_ONLY, AblationType.TAG_ONLY]
    loss_fn = [LossFunc.HUBER, LossFunc.MSE, LossFunc.LOG_CASH]

    for head in heads:
        for ablation in ablations:
            for loss_function in loss_fn:

                print(f"\n>>> Running: {head.value} | {ablation.value} | {loss_function.value}")
            
                # Reset Model and Trainer for a fresh start
                model = RecipeNet(
                    meta_in=full_dataset.meta_dim, 
                    tag_in=full_dataset.tag_dim, 
                    hidden_dim=cfg.hidden_dim, 
                    head_type=head
                )
                trainer = Trainer(model, train_loader, val_loader, cfg)
                
                # Fit 
                history = trainer.fit(epochs=cfg.epochs, head_type=head, ablation=ablation, loss_fn=loss_function)
                
                # Evaluate and capture cohesive results
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_id = f"{head.value}_{ablation.value}_{loss_function.value}_{timestamp}"
                
                test_metrics, embeds = trainer.evaluate(
                    test_loader, head, ablation, return_embeddings=cfg.return_embeddings
                )
                
                # Save cohesive files using same experiment_id
                history.update(test_metrics)
                results_path = os.path.join(s.results_dir, f"results_{experiment_id}.json")
                with open(results_path, 'w') as f:
                    json.dump(history, f, indent=4)
                
                if embeds is not None:
                    torch.save(embeds, os.path.join(s.results_dir, f"embeds_{experiment_id}.pt"))
                
                print(f"Finished {experiment_id}")
    
    

if __name__ == "__main__":
    main()