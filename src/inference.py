# src/inference.py

from pathlib import Path

import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.preprocessing import preprocess_data
from src.config import load_settings
from src.dataset import RecipeDataset
from src.models import RecipeNet, HeadType, AblationType

def run_inference(best_model_path: str, winning_head: HeadType, output_name: str = "final_corpus_embeddings.pt"):
    print(f"\n--- Starting Full Corpus Inference ---")
    
    # Load Setup & Data
    s = load_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the processed data and instantiate the dataset
    df = preprocess_data(s, overwrite_processed=False)
    full_dataset = RecipeDataset(df)
        
    inference_loader = DataLoader(
        full_dataset, 
        batch_size=1024, 
        shuffle=False
    )
    
    # Rebuild the model architecture from winning config
    print(f"Instantiating RecipeNet with {winning_head.value} head.")
    hidden_dim = 128 
    
    model = RecipeNet(
        meta_in=full_dataset.meta_dim, 
        tag_in=full_dataset.tag_dim, 
        hidden_dim=hidden_dim, 
        head_type=winning_head,
        num_meta=full_dataset.num_dim,
        cat_meta=full_dataset.cat_dim
    ).to(device)

    # Load the best model weights
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Could not find model weights at: {best_model_path}")

    print(f"Loading weights from {os.path.basename(best_model_path)}.")
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    # Initialize storage for results
    all_ids = []
    all_names = []
    all_targets = []
    all_preds = []
    all_embeddings = []

    print(f"Processing {len(full_dataset)} recipes through the inference pipeline:")

    with torch.no_grad():
        for meta_x, tag_x, targets, ids, names in tqdm(inference_loader, desc="Inference"):
            meta_x = meta_x.to(device)
            tag_x = tag_x.to(device)

            outputs, embeddings = model(
                meta_x,
                tag_x,
                return_embeddings=True,
                ablation=AblationType.ALL_FEATURES
            )

            all_preds.extend(outputs.cpu().view(-1).tolist())
            all_embeddings.append(embeddings.cpu())
            all_targets.extend(targets.cpu().view(-1).tolist())
            all_ids.extend(ids)
            all_names.extend(names)

    final_embeddings_tensor = torch.cat(all_embeddings, dim=0)

    print(f"Prediction range: [{min(all_preds):.4f}, {max(all_preds):.4f}]")

    # Bundle and Save
    inference_bundle = {
        "recipe_ids": all_ids,
        "recipe_names": all_names,
        "targets": all_targets,
        "predictions": all_preds,
        "embeddings": final_embeddings_tensor
    }
    
    output_path = os.path.join(s.results_dir, output_name)
    torch.save(inference_bundle, output_path)
    
    print(f"Saved {len(all_ids)} recipes.")
    print(f"Embeddings Shape: {final_embeddings_tensor.shape}")
    print(f"Output located at: {output_path}")

if __name__ == "__main__":
    WINNING_MODEL_FILE = "data/models/best/runs/best_model_residual_v2_all_features_mse.pth"
    WINNING_HEAD = HeadType.RESIDUAL_V2

    run_inference(
        WINNING_MODEL_FILE,
        WINNING_HEAD,
        output_name="final_residual_v2_embeddings.pt"
    )