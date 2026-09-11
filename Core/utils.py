import os
import torch

def save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, curriculum_stage, completed_stages):
    """Save training states including model, optimizer, scheduler, and curriculum status for fault tolerance."""
    print(f"Saving checkpoint to {ckpt_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)

    # Extract underlying model if wrapped in DDP
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'curriculum_stage': curriculum_stage,
        'completed_stages': completed_stages
    }
    torch.save(checkpoint, ckpt_path)
    print("Checkpoint saved.")

def load_checkpoint(ckpt_path, model, optimizer=None, scheduler=None):
    """Load model, optimizer, scheduler, and training progress from checkpoint."""
    print(f"Loading checkpoint from {ckpt_path}...")
    map_location = {'cuda:%d' % 0: f'cuda:{torch.cuda.current_device()}'} if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    
    # Backwards compatibility: If the checkpoint is just a model state dict directly
    if 'model_state_dict' not in checkpoint:
        model_state = checkpoint
        epoch, curriculum_stage, completed_stages = 0, 0, []
        has_full_state = False
    else:
        model_state = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 0)
        curriculum_stage = checkpoint.get('curriculum_stage', 0)
        completed_stages = checkpoint.get('completed_stages', [])
        has_full_state = True
        
    # Handle DDP prefix mismatch depending on how model was saved vs loaded
    from torch.nn.parallel import DistributedDataParallel as DDP
    is_ddp = isinstance(model, DDP)
    
    first_key = list(model_state.keys())[0] if model_state else ""
    if first_key.startswith('module.') and not is_ddp:
        model_state = {k[7:]: v for k, v in model_state.items()}
    elif not first_key.startswith('module.') and is_ddp:
        model_state = {'module.'+k: v for k, v in model_state.items()}
        
    model.load_state_dict(model_state)
    
    if has_full_state:
        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print("Loaded completely.")
    return epoch, curriculum_stage, completed_stages
