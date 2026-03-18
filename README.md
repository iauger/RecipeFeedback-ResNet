# Recipe Quality Modeling & Semantic Embeddings - CS 615 Final Project

This project implements neural network architectures to model recipe quality from structured metadata and review-derived signals. The primary objective is to evaluate how architectural design impacts both **predictive performance** and **embedding quality**, with downstream application to **semantic re-ranking in an information retrieval system**.

---

## Execution Pipeline

The project is designed for reproducibility and modular experimentation. The full workflow—from preprocessing to training, inference, and visualization—can be executed via the main entry point or explored through supporting scripts.

* **Entry Point:** `main.py`
* **Model Outputs:** Saved under `models/results/`
* **Visualizations:** Generated via `visualizations.py`
* **Inference Outputs:** Produced using `inference.py`
* **Execution:** Configure experiment parameters (model, ablation, loss, hyperparameters, inference) in `main.py` and run training/inference. Evaluation is currently performed in `notebooks/project_report.ipynb`. 

---

## Project Structure

Core components of the project repo

```
├───.vscode
├───data
│   ├───models
│   │   ├───best
│   │   │   └───runs
│   │   └───results
│   ├───processed
│   │   └───features
│   └───raw
│       └───gold
│           ├───gold_labeled_reviews_20260310_135905.parquet # Labeled reviews
│           ├───modeling_recipe.parquet # Cleaned recipe dataset
│           └───modeling_reviews.parquet # Cleaned, unlabeled reviews dataset
├───notebooks
│   └───project_report.ipynb
├───Reference Documents
├───main.py # entry point for training/orchestration
└───src
    ├───config.py              # Global configuration (paths, hyperparameters, enums)
    ├── dataset.py             # Dataset loading and PyTorch Dataset definitions
    ├── preprocessing.py       # Feature construction and normalization
    ├── layers.py              # Custom neural network layers (e.g., PLQP, residual blocks)
    ├── models.py              # Model architectures (MLP, Residual, Two-Tower)
    ├── trainer.py             # Training loop, evaluation, checkpointing
    ├── inference.py           # Full-corpus inference and embedding generation
    └── visualizations.py      # UMAP + diagnostic visualizations
```

---

## Project Modules

### 1. Data Processing (`preprocessing.py`)
Transforms raw data into model-ready features.
- Structured metadata (time, nutrition, ingredients)
- Aggregated review statistics
- Multi-label semantic tags  
- Standard scaling  
- Outputs train/validation feature matrices  

---

### 2. Dataset (`dataset.py`)
PyTorch dataset definitions.
- Feature tensor construction  
- Smoothed rating target  
- DataLoader support  

---

### 3. Custom Layers (`layers.py`)
Reusable model components.
- Residual blocks (skip connections)  
- PLQP layer (feature interactions)  
- Supports deep MLP and two-tower models  

---

### 4. Models (`models.py`)
Implemented architectures:
- Shallow MLP  
- Deep MLP  
- Residual MLP (Residual V2)  
- Two-Tower (metadata + review encoders)  

All models share consistent inputs and training setup.

---

### 5. Training (`trainer.py`)
Training and evaluation pipeline.
- Loss: MSE (configurable)  
- Optimizer: Adam  
- Early stopping + validation tracking  
- Saves best-performing checkpoints  

---

### 6. Inference (`inference.py`)
Full-dataset inference.
- Predicted quality scores  
- Embedding generation  
- Supports ranking and retrieval analysis  

---

### 7. Visualization (`visualizations.py`)
Model diagnostics.
- UMAP embedding projections  
- Leaderboard comparisons (models, features, loss)  

---

### 8. Configuration (`config.py`)
Centralized settings.
- Model types and ablations  
- Hyperparameters  
- File paths  