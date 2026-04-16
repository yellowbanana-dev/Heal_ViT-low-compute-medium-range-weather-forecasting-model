"""
Plot zonal energy spectra: HEAL-ViT predictions vs ERA5 ground truth.

Reads ``spectra_metrics.json`` produced by ``evaluate_spectra.py`` and
generates:
  1. Per-lead-time figures (all 20 variables, 4x5 grid) — log-log plots.
  2. A summary figure for key variables across all lead times.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_spectra(json_path="spectra_metrics.json"):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run evaluate_spectra.py first.")
        return

    with open(json_path) as f:
        data = json.load(f)

    variables  = data["metadata"]["variables"]
    steps      = data["metadata"]["spectra_steps"]
    wavenumbers = np.array(data["metadata"]["wavenumbers"])
    n_vars  = len(variables)
    n_steps = len(steps)

    # ------------------------------------------------------------------
    # 1.  One figure per lead time — all 20 variables (4x5)
    # ------------------------------------------------------------------
    for step in steps:
        pred  = np.array(data[f"pred_step_{step}"])    # (C, n_wn)
        truth = np.array(data[f"truth_step_{step}"])   # (C, n_wn)

        fig, axes = plt.subplots(4, 5, figsize=(28, 20))
        axes = axes.flatten()

        for i, var in enumerate(variables):
            ax = axes[i]
            wn = wavenumbers[1:]            # skip k=0 (the mean)
            ax.loglog(wn, truth[i, 1:], color='black', linewidth=1.5,
                      label='ERA5 (Truth)', alpha=0.8)
            ax.loglog(wn, pred[i, 1:], color='tab:red', linewidth=1.5,
                      label='HEAL-ViT', alpha=0.8)
            ax.set_title(var, fontsize=10)
            ax.grid(True, which='both', linestyle='--', alpha=0.3)
            if i >= 15:
                ax.set_xlabel('Wavenumber', fontsize=9)
            if i % 5 == 0:
                ax.set_ylabel('Spectral Energy', fontsize=9)
            if i == 0:
                ax.legend(fontsize=8)

        hours = step * 6
        fig.suptitle(
            f'Zonal Energy Spectrum — Lead Time {hours}h (Step {step})',
            fontsize=16, y=1.01,
        )
        fig.tight_layout()
        fname = f"spectra_step_{step}.png"
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {fname}")

    # ------------------------------------------------------------------
    # 2.  Summary figure — key variables across all lead times
    # ------------------------------------------------------------------
    key_vars = [
        '2m_temperature',
        'geopotential_500',
        'u_component_of_wind_500',
        'specific_humidity_850',
    ]
    key_indices = [variables.index(v) for v in key_vars if v in variables]
    key_names   = [variables[i] for i in key_indices]

    if key_indices:
        fig, axes = plt.subplots(
            len(key_indices), n_steps,
            figsize=(5 * n_steps, 4.5 * len(key_indices)),
            squeeze=False,
        )
        for row, (vi, vname) in enumerate(zip(key_indices, key_names)):
            for col, step in enumerate(steps):
                pred  = np.array(data[f"pred_step_{step}"])
                truth = np.array(data[f"truth_step_{step}"])
                ax = axes[row, col]
                wn = wavenumbers[1:]
                ax.loglog(wn, truth[vi, 1:], 'k-', lw=1.5, label='ERA5')
                ax.loglog(wn, pred[vi, 1:],  'r-', lw=1.5, label='HEAL-ViT')
                ax.set_title(f'{vname}\n{step * 6}h', fontsize=10)
                ax.grid(True, which='both', ls='--', alpha=0.3)
                if row == len(key_indices) - 1:
                    ax.set_xlabel('Wavenumber')
                if col == 0:
                    ax.set_ylabel('Spectral Energy')
                if row == 0 and col == 0:
                    ax.legend(fontsize=8)

        fig.suptitle(
            'Zonal Energy Spectrum — Key Variables',
            fontsize=16, y=1.01,
        )
        fig.tight_layout()
        fig.savefig('spectra_summary.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print("Saved spectra_summary.png")


if __name__ == "__main__":
    plot_spectra()
