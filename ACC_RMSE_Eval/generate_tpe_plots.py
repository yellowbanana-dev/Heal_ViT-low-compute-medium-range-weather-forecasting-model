import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Data_Config.config import load_config
from Data_Config.data import ERA5ZarrDataset, compute_normalization_stats
from Core.model import HEALViTModel
from Core.graph import HealpixGraphGenerator

def plot_tpe(target, pred, error, var_name, lead_time_h, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(18, 5))
    
    # Common projection
    proj = ccrs.Robinson()
    
    # Target
    ax1 = fig.add_subplot(1, 3, 1, projection=proj)
    ax1.set_global()
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax1.set_title("Target")
    vmax = max(np.abs(target).max(), np.abs(pred).max())
    vmin = min(target.min(), pred.min())
    im1 = ax1.pcolormesh(np.linspace(0, 360, 240), np.linspace(90, -90, 121), target, 
                         transform=ccrs.PlateCarree(), cmap='viridis', vmin=vmin, vmax=vmax, shading='gouraud')
    plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)

    # Prediction
    ax2 = fig.add_subplot(1, 3, 2, projection=proj)
    ax2.set_global()
    ax2.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax2.set_title("Prediction")
    im2 = ax2.pcolormesh(np.linspace(0, 360, 240), np.linspace(90, -90, 121), pred, 
                         transform=ccrs.PlateCarree(), cmap='viridis', vmin=vmin, vmax=vmax, shading='gouraud')
    plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)

    # Error
    ax3 = fig.add_subplot(1, 3, 3, projection=proj)
    ax3.set_global()
    ax3.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax3.set_title("Error (Pred - Target)")
    abs_err_max = np.abs(error).max()
    im3 = ax3.pcolormesh(np.linspace(0, 360, 240), np.linspace(90, -90, 121), error, 
                         transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-abs_err_max, vmax=abs_err_max, shading='gouraud')
    plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.8)
    
    plt.suptitle(f"{var_name} at Lead Time {lead_time_h}h", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{var_name}_{lead_time_h}h.png", dpi=200, bbox_inches='tight')
    plt.close()

def main():
    config = load_config('../config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    norm_stats = compute_normalization_stats(config)
    generator = HealpixGraphGenerator(config)
    graph_data = generator.generate_graph_data()
    
    model = HEALViTModel(config, graph_data).to(device)
    
    ckpt_path = "../healvit_wb2_ar6.pt"
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # Fix DDP keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    max_steps = 20 # 120 hours (5 days)
    
    val_dataset = ERA5ZarrDataset(config, graph_data, config.val_years[0], config.val_years[1], norm_stats, mode='val', max_ar_steps=max_steps, in_memory=True)
    # Get just one batch
    batch = val_dataset[0] # Note: ERA5ZarrDataset returns Data object for PYG
    # Actually dataloader is better
    from torch_geometric.loader import DataLoader
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)
    
    var_stds = []
    for var in config.surface_vars:
        var_stds.append(norm_stats[f"{var}_std"])
    for level in config.pressure_levels:
        for var in config.level_vars:
            var_stds.append(norm_stats[f"{var}_{level}_std"])
    
    var_means = []
    for var in config.surface_vars:
        var_means.append(norm_stats[f"{var}_mean"])
    for level in config.pressure_levels:
        for var in config.level_vars:
            var_means.append(norm_stats[f"{var}_{level}_mean"])
            
    stds_tensor = torch.tensor(var_stds, device=device).view(1, config.out_features, 1, 1)
    means_tensor = torch.tensor(var_means, device=device).view(1, config.out_features, 1, 1)

    all_vars = config.all_vars
    print("Variables:", all_vars)
    
    target_vars = ['specific_humidity_500', '2m_temperature', 'geopotential_500', 'u_component_of_wind_250', 'v_component_of_wind_250']
    var_indices = {var: all_vars.index(var) for var in target_vars}
    
    print("Running inference...")
    x_current = batch.x.clone()
    y_reshaped = batch.y.view(1, max_steps, -1, config.out_features)
    
    with torch.no_grad():
        for k in range(max_steps):
            batch.x = x_current
            pred = model(batch)
            
            # evaluate at step 20 (120h)
            if k == max_steps - 1: # 120h
                target = y_reshaped[:, k, :, :].reshape(-1, config.out_features)
                pred_spatial = pred.view(1, config.h_grid, config.w_grid, config.out_features).permute(0, 3, 1, 2)
                target_spatial = target.view(1, config.h_grid, config.w_grid, config.out_features).permute(0, 3, 1, 2)
                
                # Denormalize
                pred_phys = pred_spatial * stds_tensor + means_tensor
                target_phys = target_spatial * stds_tensor + means_tensor
                error_phys = pred_phys - target_phys
                
                pred_phys = pred_phys[0].cpu().numpy()
                target_phys = target_phys[0].cpu().numpy()
                error_phys = error_phys[0].cpu().numpy()
                
                for var, idx in var_indices.items():
                    print(f"Plotting {var}...")
                    plot_tpe(target_phys[idx], pred_phys[idx], error_phys[idx], var, (k+1)*6)
                    
            x_current = pred

if __name__ == '__main__':
    main()
