import json
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_spectra_overlay(json_6l, json_48l):
    if not os.path.exists(json_6l):
        print(f"Error: 6L json {json_6l} not found.")
        return
    if not os.path.exists(json_48l):
        print(f"Error: 48L json {json_48l} not found.")
        return

    with open(json_6l) as f:
        data_6 = json.load(f)
    with open(json_48l) as f:
        data_48 = json.load(f)

    variables  = data_6["metadata"]["variables"]
    steps      = data_6["metadata"]["spectra_steps"]
    wavenumbers = np.array(data_6["metadata"]["wavenumbers"])
    n_vars  = len(variables)
    n_steps = len(steps)

    key_vars = [
        '2m_temperature',
        'geopotential_500',
        'u_component_of_wind_500',
        'specific_humidity_850',
    ]
    key_indices = [variables.index(v) for v in key_vars if v in variables]
    key_names   = [variables[i] for i in key_indices]

    if not key_indices:
        return

    # ------------------------------------------------------------------
    # 1. Overlay Plot (Log-Log Scale)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        len(key_indices), n_steps,
        figsize=(5 * n_steps, 4.5 * len(key_indices)),
        squeeze=False,
    )
    for row, (vi, vname) in enumerate(zip(key_indices, key_names)):
        for col, step in enumerate(steps):
            pred_6  = np.array(data_6[f"pred_step_{step}"])
            pred_48 = np.array(data_48[f"pred_step_{step}"])
            truth   = np.array(data_48[f"truth_step_{step}"]) # Truth is ERA5

            ax = axes[row, col]
            wn = wavenumbers[1:]
            
            ax.loglog(wn, truth[vi, 1:], 'k-', lw=2.0, label='ERA5', alpha=0.9)
            ax.loglog(wn, pred_6[vi, 1:], '-', color='tab:blue', lw=1.5, label='6L Model', alpha=0.8)
            ax.loglog(wn, pred_48[vi, 1:], '-', color='tab:red', lw=1.5, label='48L Model', alpha=0.8)
            
            ax.set_title(f'{vname}\n{step * 6}h', fontsize=10)
            ax.grid(True, which='both', ls='--', alpha=0.3)
            
            if row == len(key_indices) - 1:
                ax.set_xlabel('Wavenumber')
            if col == 0:
                ax.set_ylabel('Spectral Energy')
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        'Zonal Energy Spectrum — 6L vs 48L vs ERA5',
        fontsize=16, y=1.01,
    )
    fig.tight_layout()
    out_name = 'spectra_overlay_summary.png'
    fig.savefig(out_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_name}")

    # ------------------------------------------------------------------
    # 2. Ratio Plot (Linear Scale on Y, Log on X)
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(
        len(key_indices), n_steps,
        figsize=(5 * n_steps, 4.5 * len(key_indices)),
        squeeze=False,
    )
    for row, (vi, vname) in enumerate(zip(key_indices, key_names)):
        for col, step in enumerate(steps):
            pred_6  = np.array(data_6[f"pred_step_{step}"])
            pred_48 = np.array(data_48[f"pred_step_{step}"])
            truth   = np.array(data_48[f"truth_step_{step}"])

            ax = axes2[row, col]
            wn = wavenumbers[1:]
            
            # Avoid division by zero
            safe_truth = np.where(truth[vi, 1:] == 0, 1e-10, truth[vi, 1:])
            ratio_6 = pred_6[vi, 1:] / safe_truth
            ratio_48 = pred_48[vi, 1:] / safe_truth

            ax.plot(wn, np.ones_like(wn), 'k--', lw=1.5, label='Optimal (ERA5 Baseline)')
            ax.plot(wn, ratio_6, '-', color='tab:blue', lw=1.5, label='6L Ratio', alpha=0.8)
            ax.plot(wn, ratio_48, '-', color='tab:red', lw=1.5, label='48L Ratio', alpha=0.8)
            
            ax.set_title(f'{vname}\n{step * 6}h', fontsize=10)
            ax.set_xscale('log')
            ax.grid(True, which='both', ls='--', alpha=0.3)
            
            if row == len(key_indices) - 1:
                ax.set_xlabel('Wavenumber')
            if col == 0:
                ax.set_ylabel('Ratio (Model / Truth)')
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
            
            # Bound Y axis to focus on the decay
            ax.set_ylim(0, 1.5)

    fig2.suptitle(
        'Spectral Energy Ratio (Model / ERA5) — 6L vs 48L',
        fontsize=16, y=1.01,
    )
    fig2.tight_layout()
    out_name2 = 'spectra_ratio_summary.png'
    fig2.savefig(out_name2, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved {out_name2}")


if __name__ == "__main__":
    path_6l = "/storage/bharat/Heal_ViT_Refactored/Power_Spectra/spectra_metrics.json"
    path_48l = "/storage/bharat/Heal_ViT_Refactored_feat_48swin/Power_Spectra/spectra_metrics.json"
    plot_spectra_overlay(path_6l, path_48l)
