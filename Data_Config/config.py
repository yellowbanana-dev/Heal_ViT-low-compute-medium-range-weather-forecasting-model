import yaml
import os

class Config:
    def __init__(self, config_dict):
        # Flattened for convenience or structured as objects
        self.data_config = config_dict.get('data', {})
        self.model_config = config_dict.get('model', {})
        self.training_config = config_dict.get('training', {})
        
        # Data
        self.zarr_path = self.data_config.get('zarr_path')
        self.h_grid = self.data_config.get('h_grid', 121)
        self.w_grid = self.data_config.get('w_grid', 240)
        self.n_grid = self.h_grid * self.w_grid
        self.train_years = self.data_config.get('train_years', [1979, 2018])
        self.val_years = self.data_config.get('val_years', [2019, 2020])
        self.surface_vars = self.data_config.get('surface_vars', [])
        self.level_vars = self.data_config.get('level_vars', [])
        self.pressure_levels = self.data_config.get('pressure_levels', [])
        
        # Derived Data Properties
        self.all_vars = self.surface_vars + [
            f"{var}_{level}" for level in self.pressure_levels for var in self.level_vars
        ]
        self.num_vars = len(self.all_vars)

        # Model
        self.in_features = self.num_vars
        self.out_features = self.num_vars
        self.edge_embed_dim = self.model_config.get('edge_embed_dim', 32)
        self.embed_dim = self.model_config.get('embed_dim', 128)
        self.mesh_dim = self.model_config.get('mesh_dim', 128)
        self.num_heads = self.model_config.get('num_heads', 4)
        self.mlp_ratio = self.model_config.get('mlp_ratio', 4.0)
        self.drop_rate = self.model_config.get('drop_rate', 0.0)
        self.num_bottleneck_layers = self.model_config.get('num_bottleneck_layers', 6)
        self.use_checkpoint = self.model_config.get('use_checkpoint', True)

        # Training
        self.per_gpu_batch_size = self.training_config.get('per_gpu_batch_size', 1)
        self.accumulation_steps = self.training_config.get('accumulation_steps', 4)
        self.use_amp = self.training_config.get('use_amp', True)
        self.base_lr = float(self.training_config.get('base_lr', 2.5e-4))
        self.finetune_lr = float(self.training_config.get('finetune_lr', 1e-7))
        self.weight_decay = self.training_config.get('weight_decay', 0.1)
        raw_curriculum = self.training_config.get('curriculum_steps', [1, 60])
        raw_epochs = self.training_config.get('epochs_per_stage', [11, 1])

        # Auto-expand rollout specification
        if len(raw_curriculum) == 2 and raw_curriculum[0] == 1 and raw_curriculum[1] > 2:
            max_steps = raw_curriculum[1]
            self.curriculum_steps = [1] + list(range(2, max_steps + 1))
            self.epochs_per_stage = [raw_epochs[0]] + [raw_epochs[1]] * (max_steps - 1)
        else:
            self.curriculum_steps = raw_curriculum
            self.epochs_per_stage = raw_epochs

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    return Config(config_dict)
