import os
import sys
import json
import torch
import numpy as np
import xarray as xr
import torch.distributed as dist
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Append parent dir so we can import modules gracefully
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data_Config.config import load_config
from Data_Config.data import ERA5ZarrDataset, compute_normalization_stats, load_climatology
from Core.model import HEALViTModel
from Core.graph import HealpixGraphGenerator
from Core.metrics import get_latitude_weights, ACCMetric
from train import setup_distributed, cleanup_distributed
from Core.paths import CONFIG_PATH, checkpoint

def run_evaluation():
    config = load_config(CONFIG_PATH)
    global_rank, local_rank, world_size = setup_distributed()
    is_master = (global_rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    max_ar_steps = 60
    
    if is_master:
        print("="*60, flush=True)
        print(f"EVALUATING {max_ar_steps} STEPS ACROSS ALL VARIABLES", flush=True)
        print("="*60, flush=True)
        
    norm_stats = compute_normalization_stats(config)
    generator = HealpixGraphGenerator(config)
    graph_data = generator.generate_graph_data()
    
    model = HEALViTModel(config, graph_data).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
    ckpt_path = checkpoint()  # final curriculum stage
    if is_master: print(f"Loading checkpoint {ckpt_path}...", flush=True)
    
    # Safely load the state dict regardless of DDP structure
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
        
    # Standardize DDP keys (if saved with DDP, keys have 'module.')
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.') and world_size == 1:
            new_state_dict[k[7:]] = v
        elif not k.startswith('module.') and world_size > 1:
            new_state_dict['module.' + k] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Load val dataset
    val_dataset = ERA5ZarrDataset(config, graph_data, config.val_years[0], config.val_years[1], norm_stats, mode='val', max_ar_steps=max_ar_steps, in_memory=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False) if world_size > 1 else None
    val_loader = DataLoader(val_dataset, batch_size=config.per_gpu_batch_size, shuffle=False, sampler=val_sampler, num_workers=0, pin_memory=True)
    
    lat_weights = get_latitude_weights(config.h_grid).to(device)
    lat_weights_acc = lat_weights.view(1, 1, config.h_grid, 1)
    acc_metric = ACCMetric().to(device)
    
    climatology_path = '/home/bedartha/public/datasets/for_model_development/weatherbench2/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative_MWE.zarr'
    climatology = load_climatology(climatology_path, config, norm_stats, device)
    
    # Extricate STDs for denormalizing RMSE later
    var_stds = []
    for var in config.surface_vars:
        var_stds.append(norm_stats[f"{var}_std"])
    for level in config.pressure_levels:
        for var in config.level_vars:
            var_stds.append(norm_stats[f"{var}_{level}_std"])
    stds_tensor = torch.tensor(var_stds, device=device).view(1, config.out_features, 1, 1)
    
    # Tracking variables logic
    # Tensors are (max_ar_steps, num_variables)
    step_loss_per_var = torch.zeros(max_ar_steps, config.out_features, device=device)
    step_acc_per_var = torch.zeros(max_ar_steps, config.out_features, device=device)
    n_batches = 0
    spatial_biases_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = batch.to(device)
            batch_size = batch.num_graphs
            y_reshaped = batch.y.view(batch_size, max_ar_steps, -1, config.out_features)
            
            x_current = batch.x.clone()
            
            for k in range(max_ar_steps):
                batch.x = x_current
                pred = model(batch)
                
                target = y_reshaped[:, k, :, :].reshape(-1, config.out_features)
                
                H_grid, W_grid = config.h_grid, config.w_grid
                pred_spatial = pred.view(batch_size, H_grid, W_grid, config.out_features).permute(0, 3, 1, 2)
                target_spatial = target.view(batch_size, H_grid, W_grid, config.out_features).permute(0, 3, 1, 2)
                
                # RMSE (Physical unit by multiplying std)
                sq_error = (pred_spatial - target_spatial) ** 2
                
                # Capture raw bias for spatial visualization (first item in batch)
                if is_master and batch_idx == 0:
                    raw_bias = ((pred_spatial - target_spatial) * stds_tensor)[0].detach().cpu().numpy()
                    spatial_biases_list.append(raw_bias)
                
                weighted_sq_error = sq_error * lat_weights_acc
                
                # Mean over Batch (0), H (2), W (3) => leaving out_features (1)
                rmse_norm_per_channel = torch.sqrt(torch.mean(weighted_sq_error, dim=(0, 2, 3)))
                rmse_physical_per_channel = rmse_norm_per_channel * stds_tensor.view(-1)
                
                step_loss_per_var[k] += rmse_physical_per_channel
                
                # ACC
                doy = batch.doy_targets.view(batch_size, max_ar_steps)[:, k].clamp(0, climatology.shape[0] - 1)
                hour = batch.hour_targets.view(batch_size, max_ar_steps)[:, k].clamp(0, climatology.shape[1] - 1)
                clim_batch = climatology[doy, hour]
                
                acc_per_channel = acc_metric(pred_spatial, target_spatial, clim_batch, weights=lat_weights_acc)
                step_acc_per_var[k] += acc_per_channel
                
                x_current = pred
                
            n_batches += 1
            if is_master and n_batches % 10 == 0:
                print(f"Evaluated {n_batches} batches...", flush=True)

    if n_batches > 0:
        step_loss_per_var = step_loss_per_var / n_batches
        step_acc_per_var = step_acc_per_var / n_batches
        
    if world_size > 1:
        dist.all_reduce(step_loss_per_var, op=dist.ReduceOp.AVG)
        dist.all_reduce(step_acc_per_var, op=dist.ReduceOp.AVG)

    if is_master:
        print("\nEvaluation Complete! Saving JSON dict to evaluate_metrics.json...", flush=True)
        # Construct metrics JSON
        # It's an array of dictionaries per step
        metrics_out = {
            "metadata": {
                "variables": config.all_vars,
                "steps": max_ar_steps
            },
            "rmse": step_loss_per_var.cpu().tolist(), # Shape [60, 20]
            "acc": step_acc_per_var.cpu().tolist()    # Shape [60, 20]
        }
        
        with open("evaluate_metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=4)
            
        # Export spatial bias maps to NetCDF
        print("Exporting spatial bias maps to spatial_biases.nc...", flush=True)
        bias_array = np.stack(spatial_biases_list, axis=0)  # [Steps, Variables, Lat, Lon]
        data_vars = {}
        for i, var_name in enumerate(config.all_vars):
            data_vars[var_name] = (["prediction_timedelta", "lat", "lon"], bias_array[:, i, :, :])
        timedeltas = [np.timedelta64(s * 6, 'h') for s in range(1, max_ar_steps + 1)]
        ds = xr.Dataset(data_vars, coords={"prediction_timedelta": timedeltas})
        ds.to_netcdf('spatial_biases.nc')
        print("Spatial biases saved successfully.", flush=True)
            
    cleanup_distributed()

if __name__ == "__main__":
    run_evaluation()
