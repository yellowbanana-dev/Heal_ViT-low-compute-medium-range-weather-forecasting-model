import os
import json
import torch
import numpy as np
import xarray as xr
from torch_geometric.data import Data, Dataset

def compute_normalization_stats(config, save_path='normalization_stats.json'):
    if os.path.exists(save_path):
        print(f"Loading normalization statistics from {save_path}...")
        with open(save_path, 'r') as f:
            return json.load(f)

    print(f"Computing normalization statistics from {config.train_years[0]}-{config.train_years[1]}...")
    ds = xr.open_zarr(config.zarr_path)
    ds = ds.sel(time=slice(f"{config.train_years[0]}-01-01", f"{config.train_years[1]}-12-31"))
    
    stats = {}
    for var in config.surface_vars:
        data = ds[var].values
        stats[f"{var}_mean"] = float(np.mean(data))
        stats[f"{var}_std"] = float(np.std(data))
        
    for level in config.pressure_levels:
        for var in config.level_vars:
            name = f"{var}_{level}"
            data = ds[var].sel(level=level).values
            stats[f"{name}_mean"] = float(np.mean(data))
            stats[f"{name}_std"] = float(np.std(data))
            
    ds.close()
    print(f"Saving normalization statistics to {save_path}...")
    with open(save_path, 'w') as f:
        json.dump(stats, f)
    return stats

def load_climatology(climatology_path, config, norm_stats, device):
    """
    Loads the WeatherBench2 climatology, normalizes it using the training statistics,
    and returns a PyTorch tensor mapped to the GPU for ultra-fast ACC computation.
    """
    print(f"Loading Climatology once to GPU from {climatology_path}")
    ds = xr.open_zarr(climatology_path)
    
    surf_tensors = []
    # 1. Process Surface Variables
    for var in config.surface_vars:
        clim_var = var
        if var == 'mean_sea_level_pressure' and 'mean_sea_level_pressure' not in ds:
            clim_var = 'surface_pressure'
            
        data_np = ds[clim_var].values
        if data_np.ndim >= 2 and data_np.shape[-2:] == (config.w_grid, config.h_grid):
            data_np = np.swapaxes(data_np, -1, -2) # Swap to (..., Lat, Lon)
            
        data_t = torch.from_numpy(data_np).float()
        mean = norm_stats[f"{var}_mean"]
        std = norm_stats[f"{var}_std"]
        data_t = (data_t - mean) / std
        surf_tensors.append(data_t)
        
    level_tensors = []
    # 2. Process Atmospheric Level Variables
    for level in config.pressure_levels:
        for var in config.level_vars:
            name = f"{var}_{level}"
            data_np = ds[var].sel(level=level).values
            if data_np.ndim >= 2 and data_np.shape[-2:] == (config.w_grid, config.h_grid):
                data_np = np.swapaxes(data_np, -1, -2)
                
            data_t = torch.from_numpy(data_np).float()
            mean = norm_stats[f"{name}_mean"]
            std = norm_stats[f"{name}_std"]
            data_t = (data_t - mean) / std
            level_tensors.append(data_t)

    # 3. Standardize dimensions before stacking
    processed_tensors = []
    for t in surf_tensors + level_tensors:
        # If the data is daily (DayOfYear, Lat, Lon), add the 6-hour dimension
        if t.ndim == 3: 
            # Unsqueeze to (DayOfYear, 1, Lat, Lon) and expand to 4 time steps
            t = t.unsqueeze(1).expand(-1, 4, -1, -1)
        elif t.ndim == 4 and t.shape[0] == 4 and t.shape[1] == 366:
            # WeatherBench ERA5-hourly arrays are natively (Hour, Day, ...)
            t = t.transpose(0, 1) # Flip to (DayOfYear, Hour, ...)
        elif t.ndim == 4 and t.shape[0] == 366 and t.shape[1] == 4:
            pass # Already (DayOfYear, Hour, ...)
        processed_tensors.append(t)
        
    # 4. Stack explicitly on the Channel dimension (dim=2)
    # Output shape will precisely be: (366, 4, Channels, Lat, Lon)
    all_data = torch.stack(processed_tensors, dim=2)
    
    print(f"Climatology Lookup Table Shape: {all_data.shape}")
    return all_data.to(device)

class ERA5ZarrDataset(Dataset):
    def __init__(self, config, graph_data, start_year, end_year, norm_stats, 
                 mode='train', max_ar_steps=4, in_memory=False):
        super().__init__()
        import pandas as pd
        self.config = config
        self.graph_data = graph_data
        self.mode = mode
        self.max_ar_steps = max_ar_steps
        self.norm_stats = norm_stats
        self.in_memory = in_memory
        
        print(f"Loading ERA5 data ({mode}): {start_year}-{end_year}, max_ar_steps={max_ar_steps}")
        ds = xr.open_zarr(config.zarr_path)
        ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        self.times = ds.time.values
        
        # Precompute dayofyear and hour indices for fast lookup
        dates = pd.DatetimeIndex(self.times)
        self.doy = np.clip(dates.dayofyear.values - 1, 0, 365) # 0 to 365
        self.hour_idx = dates.hour.values // 6 # 0, 1, 2, 3 (for 0, 6, 12, 18)

        self.ds = ds
        self.n_times = len(self.times) - max_ar_steps
        self.lat = ds.latitude.values
        self.lon = ds.longitude.values
        
        if self.in_memory:
            cache_file = f"era5_cache_{mode}_{start_year}_{end_year}.pt"
            if os.path.exists(cache_file):
                print(f"  [Cache] Loading pre-computed tensor from '{cache_file}'...", flush=True)
                self.data = torch.load(cache_file, weights_only=True)
                print(f"  [Cache] Done. Data shape: {self.data.shape}", flush=True)
            else:
                print(f"  [Pre-loading] Loading ALL {len(self.times)} timesteps into RAM...", flush=True)
                self.data = self._load_all_data_to_ram()
                torch.save(self.data, cache_file)
        else:
            self.data = None
    
    def __len__(self): return self.n_times
    
    def _normalize(self, x, name):
        mean = self.norm_stats[f"{name}_mean"]
        std = self.norm_stats[f"{name}_std"]
        return (x - mean) / std

    def _denormalize(self, x, name):
        mean = self.norm_stats[f"{name}_mean"]
        std = self.norm_stats[f"{name}_std"]
        return x * std + mean
    
    def _load_all_data_to_ram(self):
        total_steps = len(self.times)
        surf_tensors = []
        for var in self.config.surface_vars:
            data_np = self.ds[var].values 
            if data_np.shape[1:] == (self.config.w_grid, self.config.h_grid): 
                data_np = np.transpose(data_np, (0, 2, 1))
            data_np = data_np.reshape(total_steps, -1)
            data_t = torch.from_numpy(data_np).float()
            
            mean = self.norm_stats[f"{var}_mean"]
            std = self.norm_stats[f"{var}_std"]
            data_t = (data_t - mean) / std
            surf_tensors.append(data_t)
            
        level_tensors = []
        for level in self.config.pressure_levels:
            for var in self.config.level_vars:
                name = f"{var}_{level}"
                data_np = self.ds[var].sel(level=level).values
                if data_np.shape[1:] == (self.config.w_grid, self.config.h_grid):
                    data_np = np.transpose(data_np, (0, 2, 1))
                data_np = data_np.reshape(total_steps, -1)
                data_t = torch.from_numpy(data_np).float()
                
                mean = self.norm_stats[f"{name}_mean"]
                std = self.norm_stats[f"{name}_std"]
                data_t = (data_t - mean) / std
                level_tensors.append(data_t)
                
        all_data = torch.stack(surf_tensors + level_tensors, dim=2) 
        return all_data

    def get_snapshot(self, idx):
        if self.in_memory: return self.data[idx]
        surf_data = []
        for var in self.config.surface_vars:
            x_val = self.ds[var][idx].values 
            if x_val.shape == (self.config.w_grid, self.config.h_grid):
                 x_val = x_val.T
            x_val = x_val.flatten()
            x_val = self._normalize(x_val, var)
            surf_data.append(x_val)
            
        level_data = []
        for level in self.config.pressure_levels:
            for var in self.config.level_vars:
                name = f"{var}_{level}"
                x_val = self.ds[var].sel(level=level)[idx].values
                if x_val.shape == (self.config.w_grid, self.config.h_grid):
                     x_val = x_val.T
                x_val = x_val.flatten()
                x_val = self._normalize(x_val, name)
                level_data.append(x_val)
                
        all_data = np.stack(surf_data + level_data, axis=1)
        return torch.from_numpy(all_data).float()

    def set_max_ar_steps(self, max_ar_steps):
        self.max_ar_steps = max_ar_steps
        self.n_times = len(self.times) - max_ar_steps

    def __getitem__(self, idx):
        x = self.get_snapshot(idx)
        if self.in_memory:
            y = self.data[idx+1 : idx + self.max_ar_steps + 1]
        else:
            y_list = []
            for k in range(1, self.max_ar_steps + 1):
                y_list.append(self.get_snapshot(idx + k))
            y = torch.stack(y_list, dim=0) 
            
        # Target climatology indices
        doy_targets = torch.from_numpy(self.doy[idx+1 : idx + self.max_ar_steps + 1]).long()
        hour_targets = torch.from_numpy(self.hour_idx[idx+1 : idx + self.max_ar_steps + 1]).long()
        
        return Data(
            x=x, 
            y=y,
            doy_targets=doy_targets,
            hour_targets=hour_targets,
            edge_index_G2M=self.graph_data['edge_index_G2M'],
            edge_index_Mf2Mc=self.graph_data['edge_index_Mf2Mc'],
            edge_index_Mc2Mf=self.graph_data['edge_index_Mc2Mf'],
            edge_index_M2G=self.graph_data['edge_index_M2G'],
        )
