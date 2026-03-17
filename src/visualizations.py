# src/visualizations.py

from pathlib import Path
import matplotlib.pyplot as plt
import re
import seaborn as sns
import pandas as pd
import json
import glob
import os
import numpy as np
import umap
import torch
import datetime
from scipy.stats import spearmanr
from src.config import Settings, load_settings

"""
Set of visualization and helper functions for analyzing the data, experimental results, model mappings, etc. 
I built out this file mostly to keep the not
"""

def rating_distribution_plots(
    df: pd.DataFrame,
    raw_col: str = "raw_mean_rating",
    smooth_col: str = "bayesian_rating"
) -> None:
    """
    Plot raw vs. Bayesian-smoothed rating distributions using both
    log-scaled and linear-scaled y-axes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = np.array(axes) 
     
    fig.suptitle(
        "Target Variable Engineering: Raw vs. Bayesian-Smoothed Ratings",
        fontsize=18,
        fontweight="bold",
        y=1.02
    )

    plot_specs = [
        {
            "col": raw_col,
            "ax": axes[0, 0],
            "color": "lightcoral",
            "line_color": "darkred",
            "title": "Original Raw Ratings\n(Positivity Skew and Sparse-Review Volatility)",
            "xlabel": "Rating (1–5 Stars)",
            "ylabel": "Recipe Count",
            "log_y": True,
        },
        {
            "col": smooth_col,
            "ax": axes[0, 1],
            "color": "steelblue",
            "line_color": "midnightblue",
            "title": "Bayesian-Smoothed Ratings\n(More Stable Continuous Target)",
            "xlabel": "Smoothed Rating",
            "ylabel": "Recipe Count",
            "log_y": True,
        },
        {
            "col": raw_col,
            "ax": axes[1, 0],
            "color": "lightcoral",
            "line_color": "darkred",
            "title": "Original Raw Ratings\n(Linear Scale)",
            "xlabel": "Rating (1–5 Stars)",
            "ylabel": "Recipe Count",
            "log_y": False,
        },
        {
            "col": smooth_col,
            "ax": axes[1, 1],
            "color": "steelblue",
            "line_color": "midnightblue",
            "title": "Bayesian-Smoothed Ratings\n(Linear Scale)",
            "xlabel": "Smoothed Rating",
            "ylabel": "Recipe Count",
            "log_y": False,
        },
    ]

    for spec in plot_specs:
        col = spec["col"]
        ax = spec["ax"]

        sns.histplot(
            data=df,
            x=col,
            bins=50,
            kde=True,
            color=spec["color"],
            ax=ax
        )

        mean_val = df[col].mean()
        ax.axvline(
            mean_val,
            color=spec["line_color"],
            linestyle="--",
            lw=2,
            label=f"Mean: {mean_val:.2f}"
        )

        ax.set_title(spec["title"], fontsize=13)
        ax.set_xlabel(spec["xlabel"], fontsize=11)
        ax.set_ylabel(
            f"{spec['ylabel']} (log scale)" if spec["log_y"] else spec["ylabel"],
            fontsize=11
        )
        ax.set_xlim(1, 5)

        if spec["log_y"]:
            ax.set_yscale("log")

        ax.legend()

    sns.despine()
    plt.tight_layout()
    plt.show()

def get_latest_results(results_dir: str):
    # Helper function to extract the latest result file for each (head, ablation, loss) combination
    pattern = re.compile(
        r"results_(?P<head>[^_]+)_(?P<ablation>.+?)_(?P<loss>huber|mse|mae|log_cash)_(?P<time>\d{8}_\d{6})\.json"
    )
    
    latest_runs = {}

    for f in glob.glob(os.path.join(results_dir, "results_*.json")):
        match = pattern.search(os.path.basename(f))
        if match:
            key = (match.group('head'), match.group('ablation'), match.group('loss'))
            timestamp = match.group('time')
            
            if key not in latest_runs or timestamp > latest_runs[key]['time']:
                latest_runs[key] = {'path': f, 'time': timestamp}
                
    return [val['path'] for val in latest_runs.values()]

def generate_alltime_leaderboard(
    results_dir: str,
    mode: str = "best_per_model_ablation",
    sort_by: str = "test_rmse",
    ascending: bool = True,
    print_markdown: bool = True
):
    """
    Build a leaderboard from results_*.json experiment files.
    """

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    result_files = sorted(glob.glob(os.path.join(results_dir, "results_*.json")))
    if not result_files:
        raise FileNotFoundError(f"No results_*.json files found in: {results_dir}")

    records = []

    for f in result_files:
        try:
            with open(f, "r") as j:
                data = json.load(j)

            filename = os.path.basename(f)

            # Extract timestamp from filename if present
            time_match = re.search(r"(\d{8}_\d{6})\.json$", filename)
            if time_match:
                raw_time = time_match.group(1)
                dt_obj = datetime.datetime.strptime(raw_time, "%Y%m%d_%H%M%S")
                run_date = dt_obj
                run_date_str = dt_obj.strftime("%b %d, %H:%M")
            else:
                run_date = pd.NaT
                run_date_str = "N/A"

            val_loss = data.get("val_loss", [])
            train_loss = data.get("train_loss", [])
            grad_norm = data.get("grad_norm", [])

            record = {
                "run_file": filename,
                "run_date": run_date,
                "Run Date": run_date_str,
                "Architecture": data.get("model_type", "N/A"),
                "Ablation": data.get("ablation_type", "N/A"),
                "Loss": data.get("loss_type", "N/A"),
                "test_rmse": data.get("test_rmse", None),
                "test_mse": data.get("test_mse", None),
                "test_mae": data.get("test_mae", None),
                "best_val_loss": min(val_loss) if val_loss else None,
                "final_val_loss": val_loss[-1] if val_loss else None,
                "final_train_loss": train_loss[-1] if train_loss else None,
                "epochs": len(train_loss),
                "final_grad_norm": grad_norm[-1] if grad_norm else None,
            }
            records.append(record)

        except Exception as e:
            print(f"Skipping {f} due to error: {e}")

    if not records:
        raise ValueError("No valid result files could be parsed.")

    df = pd.DataFrame(records)

    # Standard rounded display columns
    metric_cols = [
        "test_rmse", "test_mse", "test_mae",
        "best_val_loss", "final_val_loss",
        "final_train_loss", "final_grad_norm"
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = df[col].round(4)

    # Keep best run per grouping
    if mode == "all_runs":
        leaderboard = df.copy()

    elif mode == "best_per_model_ablation_loss":
        idx = (
            df.groupby(["Architecture", "Ablation", "Loss"])[sort_by]
            .idxmin() if ascending
            else df.groupby(["Architecture", "Ablation", "Loss"])[sort_by].idxmax()
        )
        leaderboard = df.loc[idx].copy()

    elif mode == "best_per_model_ablation":
        idx = (
            df.groupby(["Architecture", "Ablation"])[sort_by]
            .idxmin() if ascending
            else df.groupby(["Architecture", "Ablation"])[sort_by].idxmax()
        )
        leaderboard = df.loc[idx].copy()

    else:
        raise ValueError(
            "mode must be one of: "
            "'all_runs', 'best_per_model_ablation_loss', 'best_per_model_ablation'"
        )

    leaderboard = leaderboard.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

    # Rename for cleaner report display
    leaderboard = leaderboard.rename(columns={
        "test_rmse": "RMSE",
        "test_mse": "MSE",
        "test_mae": "MAE",
        "best_val_loss": "Best Val Loss",
        "final_val_loss": "Final Val Loss",
        "final_train_loss": "Final Train Loss",
        "epochs": "Epochs",
        "final_grad_norm": "Final Grad Norm",
    })

    # Select display columns depending on mode
    display_cols = [
        "Run Date",
        "Architecture",
        "Ablation",
        "Loss",
        "RMSE",
        "MAE",
        "MSE",
        "Best Val Loss",
        "Epochs",
    ]

    if mode == "all_runs":
        display_cols.insert(0, "run_file")

    leaderboard = leaderboard[[c for c in display_cols if c in leaderboard.columns]]

    if print_markdown:
        title_map = {
            "all_runs": "=== Full Experiment Leaderboard ===",
            "best_per_model_ablation_loss": "=== Best Run per Model × Ablation × Loss ===",
            "best_per_model_ablation": "=== Best Run per Model × Ablation ===",
        }
        print(f"\n{title_map[mode]}")
        print(leaderboard.to_markdown(index=False))

def generate_hp_leaderboard(results_dir: str):
    if not os.path.exists(results_dir):
        print(f"ERROR: Directory not found: {results_dir}")
        return
    
    latest_files = get_latest_results(results_dir)
    if len(latest_files) == 0:
        latest_files = glob.glob(os.path.join(results_dir, "results_*.json"))
        if not latest_files:
            print(f"DEBUG: No results_*.json files in folder.")
            return

    records = []
    for f in latest_files:
        try:
            # --- Strict Filename Parsing ---
            basename = os.path.basename(f)
            
            # Strip 'results_' and '.json'
            if not basename.startswith("results_") or not basename.endswith(".json"):
                continue
            core_name = basename[8:-5] 
            
            # Grab just the architecture chunk before the first underscore
            grid_arch_part = core_name.split('_')[0]
            
            # Use Regex to explicitly capture the variables, ignoring internal hyphens
            match = re.match(r"(?P<arch>[^-]+)-lr(?P<lr>.+)-bs(?P<bs>.+)-wd(?P<wd>.+)", grid_arch_part)
            
            if not match:
                continue
                
            base_arch = match.group('arch')
            lr = match.group('lr')
            bs = match.group('bs')
            wd = match.group('wd')

            # --- Read JSON for Metrics ---
            with open(f, 'r') as j:
                data = json.load(j)
                
                # --- Time Parsing ---
                time_match = re.search(r"(\d{8}_\d{6})", f)
                if time_match:
                    raw_time = time_match.group(1)
                    dt_obj = datetime.datetime.strptime(raw_time, "%Y%m%d_%H%M%S")
                    run_date_str = dt_obj.strftime("%b %d, %H:%M")
                else:
                    run_date_str = "N/A"
                
                # --- Build Record ---
                records.append({
                    'Run Date': run_date_str,
                    'Architecture': base_arch,
                    'LR': lr,
                    'Batch': bs,
                    'Decay': wd,
                    'Ablation': data.get('ablation_type', 'N/A'),
                    'Loss': data.get('loss_type', 'N/A'),
                    'RMSE': round(data.get('test_rmse', 0.0), 4),
                    'MSE': round(data.get('test_mse', 0.0), 4),
                    'MAE': round(data.get('test_mae', 0.0), 4),
                    'Epochs': len(data.get('train_loss', []))
                })
                
        except Exception as e:
            print(f"ERROR: Could not read file {f}: {e}")

    # Create DataFrame and sort
    if records:
        df = pd.DataFrame(records)
        df = df.sort_values(by=['RMSE'], ascending=True).reset_index(drop=True)
        
        print("\n=== Hyperparameter Grid Search Leaderboard ===")
        print(df.to_markdown(index=False))
    else:
        print("No valid grid search records found. (All files were skipped)")
        
def compute_recipe_umap(
    bundle_path: str,
    save_path: str | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute a 2D UMAP projection from a saved embedding bundle.
    """
    bundle = torch.load(bundle_path, map_location="cpu")

    embeddings_obj = bundle["embeddings"]
    if isinstance(embeddings_obj, torch.Tensor):
        embeddings = embeddings_obj.detach().cpu().numpy()
    else:
        embeddings = np.asarray(embeddings_obj, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings to be 2D, got shape {embeddings.shape}")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    u_map = np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)

    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path_obj), u_map)

    return u_map

def plot_recipe_manifold(
    bundle_path: str,
    projection_path: str | None = None,
    save_projection: bool = False,
    annotate_extremes: bool = True,
    point_size: float = 3.0,
    alpha: float = 0.45,
    figsize: tuple[int, int] = (14, 10),
    cmap: str = "Spectral_r",
) -> np.ndarray:
    """
    Plot the full recipe manifold. If a saved projection exists, load it.
    Otherwise compute UMAP and optionally save it.
    """
    bundle = torch.load(bundle_path, map_location="cpu")

    targets = np.asarray(bundle["targets"], dtype=np.float32).reshape(-1)
    recipe_names = bundle.get("recipe_names", [f"recipe_{i}" for i in range(len(targets))])

    if projection_path is not None and Path(projection_path).exists():
        u_map = np.load(projection_path)
    else:
        save_path = projection_path if save_projection else None
        u_map = compute_recipe_umap(bundle_path=bundle_path, save_path=save_path)

    if u_map.shape[0] != len(targets):
        raise ValueError(
            f"Projection row count ({u_map.shape[0]}) does not match "
            f"target count ({len(targets)})."
        )

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        u_map[:, 0],
        u_map[:, 1],
        c=targets,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        linewidths=0,
        rasterized=True,  # helps rendering large point clouds
    )

    if annotate_extremes:
        top_indices = targets.argsort()[-5:]
        bottom_indices = targets.argsort()[:5]
        extreme_indices = np.concatenate([top_indices, bottom_indices])

        for i in extreme_indices:
            ax.annotate(
                recipe_names[i],
                (u_map[i, 0], u_map[i, 1]),
                fontsize=8,
                weight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
            )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Bayesian Rating")

    ax.set_title("Latent Recipe Space: Full-Corpus UMAP Projection", fontsize=16)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")

    # cleaner presentation
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

    return u_map

def load_inference_frame(bundle_path: str) -> pd.DataFrame:
    """
    Load a saved inference bundle and return a dataframe for evaluation.
    """
    bundle = torch.load(bundle_path, map_location="cpu")

    targets = np.asarray(bundle["targets"], dtype=np.float32).reshape(-1)
    predictions = np.asarray(bundle["predictions"], dtype=np.float32).reshape(-1)

    if len(targets) != len(predictions):
        raise ValueError(
            f"targets length ({len(targets)}) does not match "
            f"predictions length ({len(predictions)})"
        )

    df = pd.DataFrame({
        "actual": targets,
        "predicted": predictions,
    })
    df["residual"] = df["actual"] - df["predicted"]
    df["abs_error"] = np.abs(df["residual"])

    if "recipe_ids" in bundle:
        df["recipe_id"] = bundle["recipe_ids"]

    if "recipe_names" in bundle:
        df["recipe_name"] = bundle["recipe_names"]

    return df


def summarize_inference_metrics(bundle_path: str):
    """
    Return a compact one-row dataframe of evaluation metrics derived from
    predictions stored in the inference bundle.
    """
    df = load_inference_frame(bundle_path)

    mse = float(np.mean((df["actual"] - df["predicted"]) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(df["actual"] - df["predicted"])))

    summary = pd.DataFrame([{
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "Mean Actual": round(float(df["actual"].mean()), 4),
        "Mean Predicted": round(float(df["predicted"].mean()), 4),
        "Mean Residual": round(float(df["residual"].mean()), 4),
    }])

    print(summary.to_markdown(index=False))


def plot_prediction_vs_actual(bundle_path: str, figsize: tuple[int, int] = (8, 6)) -> pd.DataFrame:
    """
    Scatter plot of predicted vs. actual ratings with 45-degree reference line.
    """
    df = load_inference_frame(bundle_path)

    plt.figure(figsize=figsize)
    plt.scatter(df["actual"], df["predicted"], s=8, alpha=0.25)

    min_val = min(df["actual"].min(), df["predicted"].min())
    max_val = max(df["actual"].max(), df["predicted"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2)

    plt.title("Predicted vs. Actual Bayesian Ratings", fontsize=14)
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.tight_layout()
    plt.show()

    return df


def plot_residual_distribution(bundle_path: str, figsize: tuple[int, int] = (8, 6), bins: int = 50) -> pd.DataFrame:
    """
    Histogram of residuals (actual - predicted).
    """
    df = load_inference_frame(bundle_path)

    plt.figure(figsize=figsize)
    sns.histplot(data=df, x="residual", bins=bins, kde=True)
    plt.axvline(0, linestyle="--", linewidth=2)
    plt.title("Residual Error Distribution", fontsize=14)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Recipe Count")
    plt.tight_layout()
    plt.show()

    return df


def plot_residuals_vs_actual(bundle_path: str, figsize: tuple[int, int] = (8, 6)) -> pd.DataFrame:
    """
    Scatter plot of residuals against actual ratings to inspect systematic bias.
    """
    df = load_inference_frame(bundle_path)

    plt.figure(figsize=figsize)
    plt.scatter(df["actual"], df["residual"], s=8, alpha=0.25)
    plt.axhline(0, linestyle="--", linewidth=2)

    plt.title("Residuals vs. Actual Bayesian Ratings", fontsize=14)
    plt.xlabel("Actual Rating")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.show()

    return df


def show_largest_prediction_errors(bundle_path: str, top_n: int = 10) -> pd.DataFrame:
    """
    Return the largest absolute prediction errors for qualitative inspection.
    """
    df = load_inference_frame(bundle_path)

    cols = [c for c in ["recipe_name", "recipe_id", "actual", "predicted", "residual", "abs_error"] if c in df.columns]
    out = df.sort_values("abs_error", ascending=False)[cols].head(top_n).reset_index(drop=True)
    return out


def plot_regression_diagnostics(bundle_path: str, figsize: tuple[int, int] = (16, 5), bins: int = 50) -> pd.DataFrame:
    """
    Combined diagnostic view:
      1. Predicted vs actual
      2. Residual distribution
      3. Residuals vs actual
    """
    df = load_inference_frame(bundle_path)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes = np.array(axes)  

    # Predicted vs actual
    axes[0].scatter(df["actual"], df["predicted"], s=8, alpha=0.25)
    min_val = min(df["actual"].min(), df["predicted"].min())
    max_val = max(df["actual"].max(), df["predicted"].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2)
    axes[0].set_title("Predicted vs. Actual")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")

    # Residual distribution
    sns.histplot(data=df, x="residual", bins=bins, kde=True, ax=axes[1])
    axes[1].axvline(0, linestyle="--", linewidth=2)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")

    # Residuals vs actual
    axes[2].scatter(df["actual"], df["residual"], s=8, alpha=0.25)
    axes[2].axhline(0, linestyle="--", linewidth=2)
    axes[2].set_title("Residuals vs. Actual")
    axes[2].set_xlabel("Actual")
    axes[2].set_ylabel("Residual")

    plt.tight_layout()
    plt.show()

    return df

def plot_umap_grid():
    bundle_specs = [
        ("Shallow",     "manifold_bundle_shallow_all_features_huber_*.pt"),
        ("Deep",        "manifold_bundle_deep_all_features_mse_*.pt"),
        ("Residual",    "manifold_bundle_residual_all_features_huber_*.pt"),
        ("Residual V2", "manifold_bundle_residual_v2_all_features_mse_*.pt"),
        ("Residual V3", "manifold_bundle_residual_v3_all_features_huber_*.pt"),
        ("Two Tower",   "manifold_bundle_two_tower_all_features_huber_*.pt"),
    ]

    s = load_settings()
    runs_dir = Path(s.best_model_dir) / "runs"
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.subplots_adjust(right=0.88, wspace=0.12, hspace=0.08)
    axes = np.array(axes).flatten()

    global_vmin = float("inf")
    global_vmax = float("-inf")
    loaded = []

    # First pass: load targets and either load or compute projections
    for title, pattern in bundle_specs:
        matches = sorted(runs_dir.glob(pattern))
        if not matches:
            print(f"Missing bundle for {title}: {pattern}")
            loaded.append((title, None, None))
            continue

        bundle_path = matches[-1]
        projection_path = bundle_path.with_suffix(".npy")

        # Reuse existing compute helper
        if projection_path.exists():
            u_map = np.load(projection_path)
        else:
            u_map = compute_recipe_umap(
                bundle_path=str(bundle_path),
                save_path=str(projection_path),
                n_neighbors=30,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )

        bundle = torch.load(bundle_path, map_location="cpu")
        targets = np.asarray(bundle["targets"], dtype=np.float32).reshape(-1)

        global_vmin = min(global_vmin, float(targets.min()))
        global_vmax = max(global_vmax, float(targets.max()))

        loaded.append((title, u_map, targets))

    # Second pass: plot shared comparison grid
    scatter = None
    for ax, (title, u_map, targets) in zip(axes, loaded):
        if u_map is None:
            ax.set_title(f"{title}\n(bundle missing)")
            ax.axis("off")
            continue

        scatter = ax.scatter(
            u_map[:, 0],
            u_map[:, 1],
            c=targets,
            cmap="viridis",
            vmin=global_vmin,
            vmax=global_vmax,
            s=4,
            alpha=0.45,
            linewidths=0,
            rasterized=True,
        )

        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if scatter is not None:
        cax = fig.add_axes((0.90, 0.18, 0.018, 0.68))  # [left, bottom, width, height]
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("Bayesian Rating")

    fig.suptitle("Embedding Manifolds by Architecture (All Features)", fontsize=16, y=0.98)
    plt.show()

def calculate_spearmanr(bundle_path):
    bundle = torch.load(bundle_path)
    df = pd.DataFrame({
        "recipe_id": bundle['recipe_ids'],
        "recipe_name": bundle['recipe_names'],
        "actual": bundle['targets'],
        "predicted": bundle['predictions'],
    })
    
    corr, p_value = spearmanr(df["actual"], df["predicted"])
    
    print(f"Spearman's Rank Correlation: {corr:.4f} (p-value: {p_value:.4e})")