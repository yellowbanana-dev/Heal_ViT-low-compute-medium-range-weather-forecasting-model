import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from Core.utils import save_checkpoint, load_checkpoint
from Core.metrics import LatitudeWeightedRMSELoss, get_latitude_weights, ACCMetric
from Data_Config.data import ERA5ZarrDataset, compute_normalization_stats, load_climatology
from Core.model import HEALViTModel
from Core.graph import HealpixGraphGenerator

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        print(f"Initialized Process Group: Rank {rank}/{world_size}, Local Rank {local_rank}")
        return rank, local_rank, world_size
    else:
        print("Not using distributed training.")
        return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def validate(model, val_loader, criterion, acc_metric, climatology, lat_weights, device, out_features, max_ar_steps):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)
            batch_size = batch.num_graphs
            
            y_reshaped = batch.y.view(batch_size, max_ar_steps, -1, out_features)
            # Use only the first AR step for standard validation or loop over steps if required
            target = y_reshaped[:, 0, :, :].reshape(-1, out_features)
            loss = criterion(pred, target)
            total_loss += loss.item()
            
            # ACC Computation - Reshape pred to (Batch, out_features, Lat, Lon)
            # Graph output is flattened, shape (Batch * N_grid, out_features)
            # H_grid = 121, W_grid = 240
            H_grid, W_grid = 121, 240
            pred_spatial = pred.view(batch_size, H_grid, W_grid, out_features).permute(0, 3, 1, 2)
            target_spatial = target.view(batch_size, H_grid, W_grid, out_features).permute(0, 3, 1, 2)
            
            # Index climatology for the batch
            # batch.doy_targets is (Batch_graphs * max_ar_steps,)
            # We want step 0 for each graph in the batch
            doy = batch.doy_targets.view(batch_size, max_ar_steps)[:, 0]
            hour = batch.hour_targets.view(batch_size, max_ar_steps)[:, 0]
            
            clim_batch = climatology[doy, hour] # Shape (Batch, Channels, Lat, Lon)
            
            acc = acc_metric(pred_spatial, target_spatial, clim_batch, weights=lat_weights)
            # mean over channels
            total_acc += acc.mean().item()
            
            n_batches += 1
            
    avg_loss = torch.tensor(total_loss / n_batches if n_batches > 0 else 0.0).to(device)
    avg_acc = torch.tensor(total_acc / n_batches if n_batches > 0 else 0.0).to(device)
    
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.AVG)
        torch.distributed.all_reduce(avg_acc, op=torch.distributed.ReduceOp.AVG)
        
    return avg_loss.item(), avg_acc.item()


def validate_rollout(model, val_loader, criterion, acc_metric, climatology, lat_weights, device, out_features, max_ar_steps):
    model.eval()
    
    step_loss = torch.zeros(max_ar_steps, device=device)
    step_acc = torch.zeros(max_ar_steps, device=device)
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            batch_size = batch.num_graphs
            y_reshaped = batch.y.view(batch_size, max_ar_steps, -1, out_features)
            
            x_current = batch.x.clone()
            
            for k in range(max_ar_steps):
                batch.x = x_current
                pred = model(batch)
                
                target = y_reshaped[:, k, :, :].reshape(-1, out_features)
                loss = criterion(pred, target)
                step_loss[k] += loss.item()
                
                H_grid, W_grid = 121, 240
                pred_spatial = pred.view(batch_size, H_grid, W_grid, out_features).permute(0, 3, 1, 2)
                target_spatial = target.view(batch_size, H_grid, W_grid, out_features).permute(0, 3, 1, 2)
                
                doy = batch.doy_targets.view(batch_size, max_ar_steps)[:, k]
                hour = batch.hour_targets.view(batch_size, max_ar_steps)[:, k]
                
                # Defensive clamp to stay within climatology bounds (366 days, 4 time-slots)
                doy = doy.clamp(0, climatology.shape[0] - 1)
                hour = hour.clamp(0, climatology.shape[1] - 1)
                
                clim_batch = climatology[doy, hour]
                
                acc = acc_metric(pred_spatial, target_spatial, clim_batch, weights=lat_weights)
                step_acc[k] += acc.mean().item()
                
                x_current = pred
                
            n_batches += 1
            
    if n_batches > 0:
        step_loss = step_loss / n_batches
        step_acc = step_acc / n_batches
        
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        torch.distributed.all_reduce(step_acc, op=torch.distributed.ReduceOp.AVG)

    is_master = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
    if is_master:
        print(f"\n--- {max_ar_steps}-Step Rollout Evaluation Results ---", flush=True)
        for k in range(max_ar_steps):
            print(f"Step {k+1} ({(k+1)*6} Hours): RMSE Loss = {step_loss[k].item():.5f} | ACC = {step_acc[k].item():.5f}", flush=True)

    return step_loss, step_acc


def train_model(config):
    global_rank, local_rank, world_size = setup_distributed()
    is_master = (global_rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if is_master:
        print("=" * 60, flush=True)
        print("HEAL-ViT Training - WeatherBench 2", flush=True)
        print(f"Variables: {config.num_vars} | Grid: {config.w_grid}x{config.h_grid}", flush=True)
        print(f"Distributed: {world_size} GPUs", flush=True)
        print("=" * 60, flush=True)

    norm_stats = compute_normalization_stats(config)

    generator = HealpixGraphGenerator(config)
    graph_data = generator.generate_graph_data()

    if is_master: print("\nCreating model...", flush=True)
    model = HEALViTModel(config, graph_data).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_master:
        total_params = sum(p.numel() for p in (model.module.parameters() if isinstance(model, DDP) else model.parameters()))
        print(f"Total parameters: {total_params:,}", flush=True)
        
    train_dataset = ERA5ZarrDataset(config, graph_data, config.train_years[0], config.train_years[1], norm_stats, mode='train', max_ar_steps=1, in_memory=True)
    val_dataset = ERA5ZarrDataset(config, graph_data, config.val_years[0], config.val_years[1], norm_stats, mode='val', max_ar_steps=1, in_memory=True)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False) if world_size > 1 else None
    
    train_loader = DataLoader(train_dataset, batch_size=config.per_gpu_batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.per_gpu_batch_size, shuffle=False, sampler=val_sampler, num_workers=0, pin_memory=True)
    
    lat_weights = get_latitude_weights(config.h_grid).to(device)
    criterion = LatitudeWeightedRMSELoss(weights=lat_weights, H=config.h_grid, W=config.w_grid)
    acc_metric = ACCMetric().to(device)
    
    # Define climatology path (or load from config if available)
    climatology_path = getattr(config, 'climatology_path', '/home/bedartha/public/datasets/for_model_development/weatherbench2/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative_MWE.zarr')
    
    if is_master:
        print("\nPreparing Climatology Data for ACC...", flush=True)
        
    try:
        climatology = load_climatology(climatology_path, config, norm_stats, device)
    except Exception as e:
        if is_master:
            print(f"Warning: Could not load climatology ({e}). ACC will not be computed correctly.", flush=True)
        # Create a dummy tensor to prevent crashing if permission is denied
        climatology = torch.zeros((366, 4, config.out_features, config.h_grid, config.w_grid), device=device)
        
    # ACC expects shape (H, W) or broadcastable for lat_weights
    lat_weights_acc = lat_weights.view(1, 1, config.h_grid, 1)

    optimizer = optim.AdamW(model.parameters(), lr=config.base_lr, weight_decay=config.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp) 
    
    completed_stages = []
    latest_checkpoint = None
    
    for steps in config.curriculum_steps:
        ckpt_path = f"healvit_wb2_ar{steps}.pt"
        if os.path.exists(ckpt_path):
            latest_checkpoint = ckpt_path

    start_epoch = 0
    start_stage = 0
    
    if latest_checkpoint:
        if is_master:
            print(f"Found existing checkpoint: {latest_checkpoint}", flush=True)
        start_epoch, start_stage, completed_stages = load_checkpoint(latest_checkpoint, model, optimizer=None, scheduler=None)
        
    training_stages = [s for s in config.curriculum_steps if s <= 6]
    for stage_idx in range(start_stage, len(training_stages)):
        num_ar_steps = training_stages[stage_idx]
        
        if num_ar_steps in completed_stages:
            continue

        if is_master:
            print(f"\n{'='*60}", flush=True)
            print(f"Curriculum Stage {stage_idx+1}: {num_ar_steps} AR Steps", flush=True)
            print(f"{'='*60}", flush=True)
            
        train_dataset.set_max_ar_steps(num_ar_steps)
        val_dataset.set_max_ar_steps(num_ar_steps)
        
        if world_size > 1:
            train_sampler.set_epoch(stage_idx)
            
        is_finetuning = (num_ar_steps > 1)
        current_lr = config.finetune_lr if is_finetuning else config.base_lr

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        steps_for_stage = config.epochs_per_stage[stage_idx] * len(train_loader)
        scheduler_steps = steps_for_stage // config.accumulation_steps

        
        if is_finetuning:
            scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=scheduler_steps)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_steps)
        train_loader_iter = iter(train_loader)
        
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        for step in range(steps_for_stage):
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch = next(train_loader_iter)

            batch = batch.to(device)

            with torch.amp.autocast('cuda', enabled=config.use_amp, dtype=torch.float16):
                batch_size_current = batch.num_graphs
                x_current = batch.x.clone()
                total_loss = 0.0
                
                for k in range(num_ar_steps):
                    batch.x = x_current
                    pred_next = model(batch)
                    
                    y_reshaped = batch.y.view(batch_size_current, num_ar_steps, -1, config.out_features)
                    target = y_reshaped[:, k, :, :].reshape(-1, config.out_features)
                    
                    loss_k = criterion(pred_next, target)
                    total_loss += loss_k
                    x_current = pred_next 
                
                final_loss = total_loss / num_ar_steps
                final_loss = final_loss / config.accumulation_steps

            scaler.scale(final_loss).backward()
            
            if (step + 1) % config.accumulation_steps == 0 or (step + 1) == steps_for_stage:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Only step the scheduler if the optimizer step was not skipped
                skip_lr_sched = (scaler.get_scale() < scale_before)
                if not skip_lr_sched:
                    scheduler.step()
            
            torch.cuda.empty_cache()

            if step % 100 == 0 and is_master:
                print(f"Stage {num_ar_steps}AR | Step {step}: Loss={final_loss.item():.5f} LR={scheduler.get_last_lr()[0]:.2e}", flush=True)

        val_loss, val_acc = validate(model, val_loader, criterion, acc_metric, climatology, lat_weights_acc, device, config.out_features, num_ar_steps)
        if is_master:
            print(f"Validation Loss (Stage {num_ar_steps}AR): {val_loss:.5f} | ACC: {val_acc:.5f}", flush=True)
            completed_stages.append(num_ar_steps)
            save_checkpoint(f"healvit_wb2_ar{num_ar_steps}.pt", model, optimizer, scheduler, epoch=0, curriculum_stage=stage_idx+1, completed_stages=completed_stages)
            
    if is_master: 
        print("\n" + "="*60, flush=True)
        print("Final 60-Step Evaluation Rollout (15 Days)", flush=True)
        print("="*60, flush=True)
        
    val_dataset.set_max_ar_steps(60)
    # Recreate the val_loader (and sampler) so the DistributedSampler picks up
    # the updated dataset length after set_max_ar_steps(60).  The old sampler
    # was created when max_ar_steps=1 and can generate indices that are now
    # out-of-range, causing CUDA index-out-of-bounds on the doy/hour lookups.
    val_sampler_rollout = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False) if world_size > 1 else None
    val_loader_rollout = DataLoader(val_dataset, batch_size=config.per_gpu_batch_size, shuffle=False, sampler=val_sampler_rollout, num_workers=0, pin_memory=True)
    validate_rollout(model, val_loader_rollout, criterion, acc_metric, climatology, lat_weights_acc, device, config.out_features, max_ar_steps=60)
    
    if is_master: print("Evaluation Complete.", flush=True)
    cleanup_distributed()
