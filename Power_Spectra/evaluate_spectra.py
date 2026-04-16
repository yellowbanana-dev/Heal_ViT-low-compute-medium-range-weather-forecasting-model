"""
Evaluate Zonal Energy Spectrum: HEAL-ViT predictions vs ERA5 ground truth.

Runs an autoregressive rollout on the validation set, denormalises
predictions and targets to physical units, and computes the latitude-
averaged zonal energy spectrum at selected lead times for every variable.
Results are saved to ``spectra_metrics.json``.

Usage (via torchrun, identical to other eval scripts):
    torchrun --nnodes=1 --nproc_per_node=2 evaluate_spectra.py
"""

import os
import json
import numpy as np
import torch
import torch.distributed as dist
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data_Config.config import load_config
from Data_Config.data import ERA5ZarrDataset, compute_normalization_stats
from Core.model import HEALViTModel
from Core.graph import HealpixGraphGenerator
from train import setup_distributed, cleanup_distributed
from power_spectra import latitude_averaged_spectrum

# ---- Configuration -------------------------------------------------------
# AR steps (1-indexed) at which spectra are computed
SPECTRA_STEPS = [1, 5, 10, 20, 40, 60]

# Maximum number of validation batches (set to None for full dataset)
MAX_BATCHES = None
# ---------------------------------------------------------------------------


def run_spectra_evaluation():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config.yaml')
    config = load_config(config_path)

    global_rank, local_rank, world_size = setup_distributed()
    is_master = (global_rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    max_ar_steps = 60

    if is_master:
        print("=" * 60, flush=True)
        print("ZONAL ENERGY SPECTRUM EVALUATION", flush=True)
        print(f"  Spectra at steps : {SPECTRA_STEPS}", flush=True)
        print(f"  Max batches      : {MAX_BATCHES}", flush=True)
        print("=" * 60, flush=True)

    # ---- Setup (mirrors evaluate_variables.py) ----------------------------
    norm_stats = compute_normalization_stats(config)
    generator = HealpixGraphGenerator(config)
    graph_data = generator.generate_graph_data()

    model = HEALViTModel(config, graph_data).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    ckpt_path = "../healvit_wb2_ar6.pt"
    if is_master:
        print(f"Loading checkpoint {ckpt_path}...", flush=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get('model_state_dict', ckpt)
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

    # Dataset & loader
    val_dataset = ERA5ZarrDataset(
        config, graph_data,
        config.val_years[0], config.val_years[1],
        norm_stats, mode='val', max_ar_steps=max_ar_steps, in_memory=True,
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size,
                           rank=global_rank, shuffle=False)
        if world_size > 1 else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.per_gpu_batch_size,
        shuffle=False, sampler=val_sampler, num_workers=0, pin_memory=True,
    )

    # ---- Per-variable denormalisation constants ----------------------------
    var_stds, var_means = [], []
    for var in config.surface_vars:
        var_stds.append(norm_stats[f"{var}_std"])
        var_means.append(norm_stats[f"{var}_mean"])
    for level in config.pressure_levels:
        for var in config.level_vars:
            var_stds.append(norm_stats[f"{var}_{level}_std"])
            var_means.append(norm_stats[f"{var}_{level}_mean"])
    stds_np  = np.array(var_stds)    # (C,)
    means_np = np.array(var_means)   # (C,)

    # ---- Grid coordinates for spectra -------------------------------------
    lat = np.linspace(90, -90, config.h_grid)
    lon = np.linspace(0, 360, config.w_grid, endpoint=False)

    H, W, C = config.h_grid, config.w_grid, config.out_features
    n_wavenumber = W // 2 + 1

    # Accumulators: for each selected step, shape (C, n_wavenumber)
    pred_spectra  = {s: np.zeros((C, n_wavenumber)) for s in SPECTRA_STEPS}
    truth_spectra = {s: np.zeros((C, n_wavenumber)) for s in SPECTRA_STEPS}
    n_batches = 0

    spectra_step_set = set(SPECTRA_STEPS)

    # ---- Rollout loop -----------------------------------------------------
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if MAX_BATCHES is not None and batch_idx >= MAX_BATCHES:
                break

            batch = batch.to(device)
            batch_size = batch.num_graphs
            y_reshaped = batch.y.view(batch_size, max_ar_steps, -1, C)

            x_current = batch.x.clone()

            for k in range(max_ar_steps):
                batch.x = x_current
                pred = model(batch)
                target = y_reshaped[:, k, :, :].reshape(-1, C)

                ar_step = k + 1  # 1-indexed

                if ar_step in spectra_step_set:
                    # Reshape to (B, H, W, C) and move to CPU/numpy
                    pred_spatial   = pred.view(batch_size, H, W, C).cpu().numpy()
                    target_spatial = target.view(batch_size, H, W, C).cpu().numpy()

                    # Denormalise to physical units
                    pred_phys   = pred_spatial   * stds_np + means_np
                    target_phys = target_spatial * stds_np + means_np

                    # Compute spectrum per variable, averaged over batch
                    for c in range(C):
                        p_spec, _ = latitude_averaged_spectrum(
                            pred_phys[:, :, :, c], lat, lon,
                        )
                        t_spec, _ = latitude_averaged_spectrum(
                            target_phys[:, :, :, c], lat, lon,
                        )
                        # Average over batch dimension
                        pred_spectra[ar_step][c]  += p_spec.mean(axis=0)
                        truth_spectra[ar_step][c] += t_spec.mean(axis=0)

                x_current = pred

            n_batches += 1
            if is_master and n_batches % 5 == 0:
                print(f"  Processed {n_batches}/{MAX_BATCHES} batches...", flush=True)

    # Average over batches
    if n_batches > 0:
        for s in SPECTRA_STEPS:
            pred_spectra[s]  /= n_batches
            truth_spectra[s] /= n_batches

    # ---- Gather across DDP ranks ------------------------------------------
    if world_size > 1:
        for s in SPECTRA_STEPS:
            pt = torch.from_numpy(pred_spectra[s]).to(device)
            tt = torch.from_numpy(truth_spectra[s]).to(device)
            dist.all_reduce(pt, op=dist.ReduceOp.AVG)
            dist.all_reduce(tt, op=dist.ReduceOp.AVG)
            pred_spectra[s]  = pt.cpu().numpy()
            truth_spectra[s] = tt.cpu().numpy()

    # ---- Save results -----------------------------------------------------
    if is_master:
        out = {
            "metadata": {
                "variables": config.all_vars,
                "spectra_steps": SPECTRA_STEPS,
                "wavenumbers": list(range(n_wavenumber)),
                "max_batches_used": n_batches,
            },
        }
        for s in SPECTRA_STEPS:
            out[f"pred_step_{s}"]  = pred_spectra[s].tolist()
            out[f"truth_step_{s}"] = truth_spectra[s].tolist()

        with open("spectra_metrics.json", "w") as f:
            json.dump(out, f)

        print(f"\nSpectra saved to spectra_metrics.json  ({n_batches} batches)",
              flush=True)

    cleanup_distributed()


if __name__ == "__main__":
    run_spectra_evaluation()
