import torch
import os
import json
import datetime
from torch.utils.data import DataLoader, random_split
from src.preprocessing import preprocess_data
from src.config import load_settings
from src.dataset import RecipeDataset
from src.models import AblationType, RecipeNet, HeadType
from src.trainer import Trainer, LossFunc
from src.inference import run_inference

# Configuration
class Config:
    learning_rate = 1e-4
    weight_decay = 0.0
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
    df = preprocess_data(s, overwrite_processed=False)  # Set to True to reprocess raw data even if processed files exist
    
    # Setup Dataset and Splits
    full_dataset = RecipeDataset(df)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)  # For reproducibility
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    
    # The Experimental Matrix
    heads = [HeadType.SHALLOW, HeadType.DEEP, HeadType.RESIDUAL, HeadType.RESIDUAL_V2, HeadType.RESIDUAL_V3, HeadType.TWO_TOWER]
    ablations = [AblationType.ALL_FEATURES, AblationType.META_ONLY, AblationType.TAG_ONLY] 
    loss_functions = [LossFunc.LOG_COSH, LossFunc.HUBER, LossFunc.MSE] 

    # Adjust these flags to control which parts of the experiment to run.
    RUN_MODEL_SWEEP = True # Sweep across all combinations of heads, ablations, and loss functions
    RUN_HPARAM_SWEEP = False # Run a hyperparameter sweep for a specific head/ablation/loss combination
    RUN_FINAL_INFERENCE = False # Run final inference using the best model and save embeddings for downstream analysis
    
    if RUN_MODEL_SWEEP:
        for head in heads: 
            for ablation in ablations:
                for loss_function in loss_functions:

                    print(f"\n>>> Running: {head.value} | {ablation.value} | {loss_function.value}")
                
                    # Reset Model and Trainer for a fresh start
                    model = RecipeNet(
                        meta_in=full_dataset.meta_dim, 
                        tag_in=full_dataset.tag_dim, 
                        hidden_dim=cfg.hidden_dim, 
                        head_type=head,
                        num_meta=full_dataset.num_dim,
                        cat_meta=full_dataset.cat_dim
                    )
                    
                    start_time = datetime.datetime.now()
                    
                    trainer = Trainer(model, train_loader, val_loader, cfg)
                    
                    # Fit 
                    history = trainer.fit(epochs=cfg.epochs, head_type=head, ablation=ablation, loss_fn=loss_function)
                    
                    # Evaluate and capture cohesive results
                    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
                    experiment_id = f"{head.value}_{ablation.value}_{loss_function.value}_{timestamp}"
                    
                    test_metrics, bundle = trainer.evaluate(
                        test_loader, head, ablation, return_embeddings=cfg.return_embeddings
                    )
                                        
                    # Save cohesive files using same experiment_id
                    history.update(test_metrics)
                    results_path = os.path.join(s.results_dir, f"results_{experiment_id}.json")
                    with open(results_path, 'w') as f:
                        json.dump(history, f, indent=4)
                    
                    if bundle is not None:
                        torch.save(bundle, os.path.join(s.results_dir, f"manifold_bundle_{experiment_id}.pt"))
                    
                    print(f"Finished {experiment_id}")
    
    if RUN_HPARAM_SWEEP:
        head = HeadType.DEEP 
        ablation = AblationType.ALL_FEATURES 
        loss_fn = LossFunc.HUBER
        learning_rates = [1e-3, 5e-4, 1e-4]
        weight_decays = [1e-4, 1e-5, 0.0]
        batch_sizes = [128, 256, 512]
        cfg = Config()  # Reset to default config for hyperparameter search
        
        for bs in batch_sizes:
            for lr in learning_rates:
                for wd in weight_decays:
                    # Create the embedded string name for the leaderboard
                    grid_arch_name = f"{head.value}-lr{lr}-bs{bs}-wd{wd}"
                    print(f"\n--- Starting Grid Run: {grid_arch_name} ---")
                    
                    # Temporarily override the config attributes for this specific loop
                    cfg.batch_size = bs
                    cfg.learning_rate = lr
                    cfg.weight_decay = wd

                    # Re-instantiate DataLoaders with the new batch size
                    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
                    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
                    
                    # Reset Model and Trainer for a fresh start
                    model = RecipeNet(
                        meta_in=full_dataset.meta_dim, 
                        tag_in=full_dataset.tag_dim, 
                        hidden_dim=cfg.hidden_dim, 
                        head_type=head,
                        num_meta=full_dataset.num_dim,
                        cat_meta=full_dataset.cat_dim
                    )
                    
                    start_time = datetime.datetime.now()
                    
                    trainer = Trainer(model, train_loader, val_loader, cfg)
                    
                    # Fit 
                    history = trainer.fit(epochs=cfg.epochs, head_type=head, ablation=ablation, loss_fn=loss_fn)
                    
                    # Evaluate and capture cohesive results
                    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
                    experiment_id = f"{grid_arch_name}_{ablation.value}_{loss_fn.value}_{timestamp}"
                    
                    test_metrics, bundle = trainer.evaluate(
                        test_loader, head, ablation, return_embeddings=cfg.return_embeddings
                    )
                                        
                    # Save cohesive files using same experiment_id
                    history.update(test_metrics)
                    results_path = os.path.join(s.results_dir, f"results_{experiment_id}.json")
                    with open(results_path, 'w') as f:
                        json.dump(history, f, indent=4)
                    
                    if bundle is not None:
                        # This now contains both 'embeddings' and 'targets'
                        torch.save(bundle, os.path.join(s.results_dir, f"manifold_bundle_{experiment_id}.pt"))
                    
                    print(f"Finished {experiment_id}")
    
    if RUN_FINAL_INFERENCE:
        best_model_path = s.best_model_path
        winning_head = HeadType.RESIDUAL_V2
        output_name = f"final_{winning_head.value.lower()}_embeddings.pt"
        run_inference(best_model_path, winning_head, output_name=output_name)
        
        
        
    

if __name__ == "__main__":
    main()