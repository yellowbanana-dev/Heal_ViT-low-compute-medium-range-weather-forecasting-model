"""
Plot Target / Prediction / Error panels for selected variables
using Cartopy Robinson projection with coastlines.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PRETTY = {
    'specific_humidity_500': 'Specific Humidity (500 hPa)',
    '2m_temperature':       '2m Temperature',
    'geopotential_500':     'Geopotential (500 hPa)',
    'u_component_of_wind_250': 'U-Wind (250 hPa)',
    'v_component_of_wind_250': 'V-Wind (250 hPa)',
}

UNITS = {
    'specific_humidity_500': 'kg/kg',
    '2m_temperature':       'K',
    'geopotential_500':     'm²/s²',
    'u_component_of_wind_250': 'm/s',
    'v_component_of_wind_250': 'm/s',
}


def plot_tpe(target, pred, error, var_name, lead_time_h, save_dir='plots_data'):
    """Plot Target, Prediction, Error on Robinson projections with coastlines."""
    os.makedirs(save_dir, exist_ok=True)

    lon = np.linspace(0, 360 - 360/240, 240)
    lat = np.linspace(90, -90, 121)

    proj = ccrs.Robinson()
    data_crs = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        1, 3, figsize=(22, 6),
        subplot_kw={'projection': proj},
    )

    display = PRETTY.get(var_name, var_name)
    unit = UNITS.get(var_name, '')

    # Shared colour range for target & prediction
    vmin = min(target.min(), pred.min())
    vmax = max(target.max(), pred.max())

    panels = [
        (axes[0], target, 'Target (ERA5)',      'viridis', vmin, vmax),
        (axes[1], pred,   'Prediction (HEAL-ViT)', 'viridis', vmin, vmax),
    ]

    # Error gets a symmetric diverging range
    emax = np.abs(error).max()
    panels.append(
        (axes[2], error, 'Error (Pred − Target)', 'RdBu_r', -emax, emax)
    )

    for ax, data, title, cmap, lo, hi in panels:
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.7, color='#333')
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.3,
                       linestyle=':', color='#666')
        ax.gridlines(draw_labels=False, linewidth=0.4, color='gray',
                     alpha=0.4, linestyle='--')

        im = ax.pcolormesh(
            lon, lat, data,
            transform=data_crs,
            cmap=cmap,
            vmin=lo, vmax=hi,
            shading='auto',
        )
        cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                          pad=0.06, shrink=0.75, aspect=30)
        cb.set_label(f'[{unit}]', fontsize=11)
        cb.ax.tick_params(labelsize=9)
        ax.set_title(title, fontsize=14, pad=10)

    fig.suptitle(
        f'HEAL-ViT (48 SWIN layers)  —  {display}\n'
        f'Lead time = {lead_time_h}h ({lead_time_h/24:.0f} days)',
        fontsize=17, y=1.04,
    )
    plt.tight_layout()

    out = os.path.join(save_dir, f'{var_name}_{lead_time_h}h.png')
    fig.savefig(out, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out}')


def main():
    for lead_h in [72, 120]:
        npz = f'plots_data/tpe_{lead_h}h.npz'
        print(f'Loading {npz} …')
        data = np.load(npz)

        for var in PRETTY:
            key = f'{var}_target'
            if key in data:
                plot_tpe(
                    data[f'{var}_target'],
                    data[f'{var}_pred'],
                    data[f'{var}_error'],
                    var, lead_h,
                )

    print('\nDone!')


if __name__ == '__main__':
    main()
