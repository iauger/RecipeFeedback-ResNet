# src/visualizations.py

import matplotlib.pyplot as plt
import re
import seaborn as sns
import pandas as pd
import json
import glob
import os
import numpy as np
from tabulate import tabulate

def get_latest_results(results_dir: str):
    # Fixed Regex: Using non-greedy match '.+?' to correctly separate 
    # underscore-heavy names like 'all_features' from 'log_cash'
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
    

def plot_faceted_comparisons(results_dir):
    latest_files = get_latest_results(results_dir)
    losses = ['huber', 'mse', 'log_cash'] 
    ablations = ['all_features', 'meta_only', 'tag_only']
    metrics = ['train_loss', 'val_loss', 'grad_norm']
    
    for l_type in losses:
        try:
            # Only create a figure if we actually have files for this loss
            relevant_files = [f for f in latest_files if f"_{l_type}_" in f]
            if not relevant_files:
                continue

            fig, axes = plt.subplots(3, 3, figsize=(18, 14))
            axes = np.array(axes).reshape(3, 3) 
            fig.suptitle(f"Experimental Matrix: {l_type.upper()} Loss Optimization", fontsize=16)
            
            for row, ablation in enumerate(ablations):
                for col, metric in enumerate(metrics):
                    ax = axes[row, col]
                     # For easier indexing
                    # Specific filter for the triplet
                    plot_files = [f for f in relevant_files if ablation in f]
                    
                    for f in plot_files:
                        with open(f, 'r') as j:
                            data = json.load(j)
                            ax.plot(data[metric], label=data['model_type'])
                    
                    ax.set_title(f"{ablation.replace('_', ' ').title()} | {metric.replace('_', ' ').title()}")
                    ax.set_xlabel("Epochs" if metric != 'grad_norm' else "Batches")
                    ax.set_ylabel(metric.replace('_', ' ').title())
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            plt.show()
        except Exception as e:
            print(f"Error plotting {l_type}: {e}")
            plt.close() # Prevent resource leaks
    
    return None

def generate_leaderboard(results_dir: str):
    
    # 1. Check if directory even exists
    if not os.path.exists(results_dir):
        print(f"ERROR: Directory not found: {results_dir}")
        return
    
    # 2. Get the files
    latest_files = get_latest_results(results_dir)
    
    if len(latest_files) == 0:
        # If this prints, your regex in get_latest_results is the problem
        all_jsons = glob.glob(os.path.join(results_dir, "results_*.json"))
        print(f"DEBUG: Total results_*.json files in folder: {len(all_jsons)}")
        if all_jsons:
            print(f"DEBUG: Example filename: {os.path.basename(all_jsons[0])}")
        return

    records = []
    for f in latest_files:
        try:
            with open(f, 'r') as j:
                data = json.load(j)
                records.append({
                    'Architecture': data.get('model_type', 'N/A'),
                    'Ablation': data.get('ablation_type', 'N/A'),
                    'Loss': data.get('loss_type', 'N/A'),
                    'RMSE': round(data.get('test_rmse', 0.0), 4),
                    'Epochs': len(data.get('train_loss', []))
                })
        except Exception as e:
            print(f"ERROR: Could not read file {f}: {e}")
        
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by=["RMSE"])
        print("\n=== Model Performance Leaderboard ===")
        print(tabulate(df.values.tolist(), headers=list(df.columns), tablefmt='pipe'))
    else:
        print("DEBUG: Records list is empty after processing files.")
    
    return df
