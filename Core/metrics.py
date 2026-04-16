import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

def get_latitude_weights(H):
    """Get latitude-based weights for grid."""
    lat = np.linspace(90, -90, H)
    delta = (lat[0] - lat[1]) / 2.0
    lat_bounds_upper_rad = np.deg2rad(lat + delta)
    lat_bounds_lower_rad = np.deg2rad(lat - delta)
    weights = np.abs(np.sin(lat_bounds_upper_rad) - np.sin(lat_bounds_lower_rad))
    weights = weights / weights.mean()
    return torch.from_numpy(weights).float().reshape(1, H, 1)

class LatitudeWeightedRMSELoss(nn.Module):
    """Area-weighted RMSE Loss."""
    def __init__(self, weights, H, W):
        super().__init__()
        self.register_buffer("weights", weights)
        self.H_grid, self.W_grid = H, W

    def forward(self, pred, target):
        B = pred.shape[0] // (self.H_grid * self.W_grid)
        C_out = pred.shape[1]
        pred_grid = pred.view(B, self.H_grid, self.W_grid, C_out)
        target_grid = target.view(B, self.H_grid, self.W_grid, C_out)
        
        sq_error = (pred_grid - target_grid) ** 2
        weighted_sq_error = sq_error * self.weights.unsqueeze(-1)
        return torch.sqrt(torch.mean(weighted_sq_error))

from ACC_RMSE_Eval.acc import calculate_acc, ACCMetric

class SpectraMetric(nn.Module):
    def __init__(self):
        super().__init__()
    # TODO: Implement Energy Spectra Metric
