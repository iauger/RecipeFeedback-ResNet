# src/visualizations.py

import matplotlib.pyplot as plt
import re
import seaborn as sns
import pandas as pd
import json
import glob
import os

def get_latest_results(results_dir: str):
    """
    Filters the results directory to return only the most recent run for each permutation of model head and ablation type
    """
    # Pattern to extract head_type and ablation_type from: results_head_ablation_timestamp.json
    pattern = re.compile(r"results_(?P<head>[^_]+)_(?P<ablation>.+)_(?P<time>\d{8}_\d{6})\.json")
    
    latest_runs = {}

    for f in glob.glob(os.path.join(results_dir, "results_*.json")):
        match = pattern.search(os.path.basename(f))
        if match:
            # Create a unique key for the experiment type
            key = (match.group('head'), match.group('ablation'))
            timestamp = match.group('time')
            
            # Keep only the one with the maximum timestamp
            if key not in latest_runs or timestamp > latest_runs[key]['time']:
                latest_runs[key] = {'path': f, 'time': timestamp}
                
    return [val['path'] for val in latest_runs.values()]
    
    
def plot_experiment_results(results_dir: str):
    # Find the latest results for each head type
    files = get_latest_results(results_dir)
    
    plt.figure(figsize=(15, 5))
    
    # Training Loss Comparison
    plt.subplot(1, 2, 1)
    for f in files:
        with open(f, 'r') as j:
            data = json.load(j)
            label = f"{data['model_type']} ({data['ablation_type']})"
            plt.plot(data['train_loss'], label=f"Train: {label}")
            plt.plot(data['val_loss'], '--', label=f"Val: {label}")
    
    plt.title("Learning Curves: Train vs Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.legend()

    # Gradient Norm Comparison
    plt.subplot(1, 2, 2)
    for f in files:
        with open(f, 'r') as j:
            data = json.load(j)
            plt.plot(data['grad_norm'], label=data['model_type'])
            
    plt.title("Optimization Stability: Gradient Norms")
    plt.xlabel("Epoch")
    plt.ylabel("Norm Magnitude")
    plt.legend()
    
    plt.tight_layout()
    plt.show()