import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import torch
import torch.distributed as dist
from torch_geometric.loader import DataLoader
from Core.utils import load_checkpoint
from Core.metrics import LatitudeWeightedRMSELoss, get_latitude_weights
from Data_Config.data import ERA5ZarrDataset, compute_normalization_stats
from Core.model import HEALViTModel
from Core.graph import HealpixGraphGenerator
from Data_Config.config import load_config

def setup_evaluation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def evaluate_rollout(config, checkpoint_path, max_ar_steps=60):
    device = setup_evaluation()
    
    # 1. Setup Data and Graph
    norm_stats = compute_normalization_stats(config)
    generator = HealpixGraphGenerator(config)
    graph_data = generator.generate_graph_data()
    
    # 2. Setup Model
    print("\nLoading model...", flush=True)
    model = HEALViTModel(config, graph_data).to(device)
    
    # 3. Load Checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found! Evaluation will use untrained weights.", flush=True)

    model.eval()

    # 4. Setup Evaluation Dataset
    print(f"\nSetting up evaluation dataset for {max_ar_steps} AR steps...", flush=True)
    # Using val_years for evaluation
    val_dataset = ERA5ZarrDataset(config, graph_data, config.val_years[0], config.val_years[1], norm_stats, mode='val', max_ar_steps=max_ar_steps, in_memory=True)
    
    # We use batch_size 1 to avoid excessive memory usage during 60-step rollout
    eval_batch_size = 1
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    
    lat_weights = get_latitude_weights(config.h_grid).to(device)
    criterion = LatitudeWeightedRMSELoss(weights=lat_weights, H=config.h_grid, W=config.w_grid)
    
    step_losses = torch.zeros(max_ar_steps).to(device)
    n_batches = 0
    
    print("\nStarting evaluation rollout...", flush=True)
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            batch_size_current = batch.num_graphs
            x_current = batch.x.clone()
            
            for k in range(max_ar_steps):
                batch.x = x_current
                # 1-step prediction
                pred_next = model(batch)
                
                # Fetch target for step k
                y_reshaped = batch.y.view(batch_size_current, max_ar_steps, -1, config.out_features)
                target = y_reshaped[:, k, :, :].reshape(-1, config.out_features)
                
                loss_k = criterion(pred_next, target)
                step_losses[k] += loss_k.item()
                
                # Update input for next step
                x_current = pred_next
                
            n_batches += 1
            if n_batches % 10 == 0:
                print(f"Processed {n_batches} eval sequences.", flush=True)
                
    # Average over batches
    step_losses = step_losses / n_batches
    
    print("\n" + "="*60)
    print(f"Evaluation Rollout Results (Averaged over {n_batches} sequences)")
    print("="*60)
    for k in range(max_ar_steps):
        print(f"Step {k+1:02d} (+{6*(k+1):02d}h): RMSE Loss = {step_losses[k].item():.5f}")

if __name__ == "__main__":
    config = load_config("config.yaml")
    checkpoint_file = "healvit_wb2_ar1.pt"
    evaluate_rollout(config, checkpoint_file, max_ar_steps=60)
