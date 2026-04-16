"""
Plot HEAL-ViT Spatial Bias Maps from spatial_biases.nc
Generates WeatherBench-style Robinson projection maps showing
spatial prediction errors (Pred - Truth).

Red  → model predicted too HIGH
Blue → model predicted too LOW
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os

# --- Physical unit labels for each variable ---
UNIT_LABELS = {
    '2m_temperature': 'K',
    '10m_u_component_of_wind': 'm/s',
    '10m_v_component_of_wind': 'm/s',
    'mean_sea_level_pressure': 'Pa',
    'total_column_water_vapour': 'kg/m²',
    'temperature_850': 'K',
    'geopotential_850': 'm²/s²',
    'specific_humidity_850': 'kg/kg',
    'u_component_of_wind_850': 'm/s',
    'v_component_of_wind_850': 'm/s',
    'temperature_500': 'K',
    'geopotential_500': 'm²/s²',
    'specific_humidity_500': 'kg/kg',
    'u_component_of_wind_500': 'm/s',
    'v_component_of_wind_500': 'm/s',
    'temperature_250': 'K',
    'geopotential_250': 'm²/s²',
    'specific_humidity_250': 'kg/kg',
    'u_component_of_wind_250': 'm/s',
    'v_component_of_wind_250': 'm/s',
}

# --- Pretty display names ---
DISPLAY_NAMES = {
    '2m_temperature': '2m Temperature',
    '10m_u_component_of_wind': '10m U-Wind',
    '10m_v_component_of_wind': '10m V-Wind',
    'mean_sea_level_pressure': 'Mean Sea Level Pressure',
    'total_column_water_vapour': 'Total Column Water Vapour',
    'temperature_850': 'Temperature (850 hPa)',
    'geopotential_850': 'Geopotential (850 hPa)',
    'specific_humidity_850': 'Specific Humidity (850 hPa)',
    'u_component_of_wind_850': 'U-Wind (850 hPa)',
    'v_component_of_wind_850': 'V-Wind (850 hPa)',
    'temperature_500': 'Temperature (500 hPa)',
    'geopotential_500': 'Geopotential (500 hPa)',
    'specific_humidity_500': 'Specific Humidity (500 hPa)',
    'u_component_of_wind_500': 'U-Wind (500 hPa)',
    'v_component_of_wind_500': 'V-Wind (500 hPa)',
    'temperature_250': 'Temperature (250 hPa)',
    'geopotential_250': 'Geopotential (250 hPa)',
    'specific_humidity_250': 'Specific Humidity (250 hPa)',
    'u_component_of_wind_250': 'U-Wind (250 hPa)',
    'v_component_of_wind_250': 'V-Wind (250 hPa)',
}


def load_and_assign_coords(nc_path):
    """Load the spatial_biases.nc and assign lat/lon coordinate values."""
    ds = xr.open_dataset(nc_path)
    
    # The grid is 121 lat × 240 lon at 1.5° resolution (WeatherBench 2 convention)
    # lat: 90°N to 90°S,  lon: 0° to 358.5°E
    nlat, nlon = ds.sizes['lat'], ds.sizes['lon']
    lat_vals = np.linspace(90, -90, nlat)
    lon_vals = np.linspace(0, 360 - (360 / nlon), nlon)
    
    ds = ds.assign_coords(lat=lat_vals, lon=lon_vals)
    return ds


def plot_single_bias_map(ds, variable, lead_time_hours, save_dir=None):
    """
    Plot a single bias map for a given variable and lead time.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset with lat/lon coordinates assigned.
    variable : str
        Variable name (e.g. 'geopotential_500').
    lead_time_hours : int or float
        Lead time in hours (e.g. 120 for 5 days).
    save_dir : str or None
        Directory to save PNG. If None, just shows interactively.
    """
    print(f"Plotting {variable} at {lead_time_hours}h lead time...")
    
    lead_time = np.timedelta64(int(lead_time_hours), 'h')
    
    if variable not in ds.data_vars:
        print(f"Error: Variable '{variable}' not found.")
        print(f"Available: {list(ds.data_vars)}")
        return
    
    try:
        data_slice = ds[variable].sel(prediction_timedelta=lead_time)
    except KeyError:
        print(f"Error: Lead time {lead_time_hours}h not found in prediction_timedelta.")
        avail_hours = ds.coords['prediction_timedelta'].values / np.timedelta64(1, 'h')
        print(f"Available lead times (hours): {avail_hours}")
        return
    
    # --- Set up Robinson projection ---
    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.Robinson())
    
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, color='#333333')
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.3, linestyle=':', color='#666666')
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    # --- Symmetric diverging colorbar centered at 0 ---
    vmax = float(np.abs(data_slice).max().values)
    vmin = -vmax
    
    # --- Plot ---
    plot = data_slice.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
        shading='gouraud',
        infer_intervals=False
    )
    
    # --- Colorbar ---
    unit = UNIT_LABELS.get(variable, '')
    cbar = plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.08, shrink=0.7)
    cbar.set_label(f'Bias [{unit}]', fontsize=12)
    
    # --- Title ---
    display = DISPLAY_NAMES.get(variable, variable)
    days = lead_time_hours / 24
    plt.title(
        f"HEAL-ViT Spatial Bias: {display}\n"
        f"Lead Time = {lead_time_hours}h ({days:.1f} days)",
        fontsize=16, pad=15,
    )
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"bias_{variable}_{int(lead_time_hours)}h.png"
        fpath = os.path.join(save_dir, fname)
        fig.savefig(fpath, dpi=400, bbox_inches='tight')
        print(f"  Saved: {fpath}")
        plt.close(fig)
    else:
        plt.show()


def plot_multi_lead(ds, variable, lead_times_hours, save_dir=None):
    """
    Plot a row of bias maps for one variable at multiple lead times.
    """
    n = len(lead_times_hours)
    fig, axes = plt.subplots(
        1, n, figsize=(6 * n, 5),
        subplot_kw={'projection': ccrs.Robinson()},
    )
    if n == 1:
        axes = [axes]
    
    unit = UNIT_LABELS.get(variable, '')
    display = DISPLAY_NAMES.get(variable, variable)
    
    # Compute a shared vmax across all requested lead times
    global_vmax = 0
    slices = []
    for lth in lead_times_hours:
        lt = np.timedelta64(int(lth), 'h')
        s = ds[variable].sel(prediction_timedelta=lt)
        slices.append(s)
        global_vmax = max(global_vmax, float(np.abs(s).max().values))
    
    for ax, s, lth in zip(axes, slices, lead_times_hours):
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, color='#333333')
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.2, linestyle=':', color='#666666')
        ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.4, linestyle='--')
        
        p = s.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r',
            vmin=-global_vmax,
            vmax=global_vmax,
            add_colorbar=False,
            shading='gouraud',
            infer_intervals=False
        )
        days = lth / 24
        ax.set_title(f"{lth}h ({days:.0f}d)", fontsize=13)
    
    # Shared colorbar
    cbar = fig.colorbar(p, ax=axes, orientation='horizontal', pad=0.15, shrink=0.5, fraction=0.046)
    cbar.set_label(f'{display} Bias [{unit}]', fontsize=12)
    
    fig.suptitle(f"HEAL-ViT Spatial Bias: {display}", fontsize=16, y=1.02)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"bias_{variable}_multi_lead.png"
        fpath = os.path.join(save_dir, fname)
        fig.savefig(fpath, dpi=400, bbox_inches='tight')
        print(f"  Saved: {fpath}")
        plt.close(fig)
    else:
        plt.show()


# ========================================================================
# Main execution
# ========================================================================
if __name__ == '__main__':
    NC_PATH = 'spatial_biases.nc'
    SAVE_DIR = 'bias_plots'
    
    ds = load_and_assign_coords(NC_PATH)
    
    # --- Key variables to plot ---
    key_vars = [
        'geopotential_500',
        '2m_temperature',
        'mean_sea_level_pressure',
        'specific_humidity_500',
        'u_component_of_wind_500',
        'temperature_850',
    ]
    
    # 1) Single maps at 5-day (120h) and 10-day (240h) lead times
    for var in key_vars:
        plot_single_bias_map(ds, var, lead_time_hours=120, save_dir=SAVE_DIR)
        plot_single_bias_map(ds, var, lead_time_hours=240, save_dir=SAVE_DIR)
    
    # 2) Multi-lead-time comparison panels for Z500 and T2m
    plot_multi_lead(ds, 'geopotential_500',
                    lead_times_hours=[24, 72, 120, 240, 360],
                    save_dir=SAVE_DIR)
    plot_multi_lead(ds, '2m_temperature',
                    lead_times_hours=[24, 72, 120, 240, 360],
                    save_dir=SAVE_DIR)
    
    print("\nDone! All plots saved to:", SAVE_DIR)
