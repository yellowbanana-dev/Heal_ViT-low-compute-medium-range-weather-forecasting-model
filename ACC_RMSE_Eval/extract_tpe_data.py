import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Data_Config.config import load_config
from Data_Config.data import ERA5ZarrDataset, compute_normalization_stats
from Core.model import HEALViTModel
from Core.graph import HealpixGraphGenerator
from Core.paths import CONFIG_PATH, checkpoint

def main():
    config = load_config(CONFIG_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    norm_stats = compute_normalization_stats(config)
    generator = HealpixGraphGenerator(config)
    graph_data = generator.generate_graph_data()
    
    model = HEALViTModel(config, graph_data).to(device)
    
    ckpt_path = checkpoint()
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    max_steps = 20  # 120 hours (5 days)
    # Steps of interest: step 12 = 72h (3 days), step 20 = 120h (5 days)
    save_steps = {11: 72, 19: 120}  # 0-indexed step -> lead time in hours
    
    from torch_geometric.loader import DataLoader
    val_dataset = ERA5ZarrDataset(config, graph_data, config.val_years[0], config.val_years[1], norm_stats, mode='val', max_ar_steps=max_steps, in_memory=True)
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
    
    os.makedirs('plots_data', exist_ok=True)
    
    with torch.no_grad():
        for k in range(max_steps):
            batch.x = x_current
            pred = model(batch)
            
            if k in save_steps:
                lead_h = save_steps[k]
                print(f"  Saving step {k+1} = {lead_h}h ...")
                target = y_reshaped[:, k, :, :].reshape(-1, config.out_features)
                pred_spatial = pred.view(1, config.h_grid, config.w_grid, config.out_features).permute(0, 3, 1, 2)
                target_spatial = target.view(1, config.h_grid, config.w_grid, config.out_features).permute(0, 3, 1, 2)
                
                pred_phys = pred_spatial * stds_tensor + means_tensor
                target_phys = target_spatial * stds_tensor + means_tensor
                error_phys = pred_phys - target_phys
                
                pred_np = pred_phys[0].cpu().numpy()
                target_np = target_phys[0].cpu().numpy()
                error_np = error_phys[0].cpu().numpy()
                
                npz_dict = {}
                for var, idx in var_indices.items():
                    npz_dict[f"{var}_target"] = target_np[idx]
                    npz_dict[f"{var}_pred"] = pred_np[idx]
                    npz_dict[f"{var}_error"] = error_np[idx]
                
                np.savez(f'plots_data/tpe_{lead_h}h.npz', **npz_dict)
                print(f"  Saved plots_data/tpe_{lead_h}h.npz")
                    
            x_current = pred
    
    print("Done!")

if __name__ == '__main__':
    main()
