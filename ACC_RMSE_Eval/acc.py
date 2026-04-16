import torch
import torch.nn as nn

def calculate_acc(pred: torch.Tensor, target: torch.Tensor, climatology: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    Calculates the Anomaly Correlation Coefficient (ACC) natively in PyTorch.
    
    Args:
        pred (torch.Tensor): Predicted tensor of shape (Batch, Channels, Lat, Lon).
        target (torch.Tensor): Ground truth tensor of shape (Batch, Channels, Lat, Lon).
        climatology (torch.Tensor): Climatology tensor of shape (Batch, Channels, Lat, Lon) 
                                    or broadcastable to it (e.g., (1, Channels, Lat, Lon)).
        weights (torch.Tensor, optional): Spatial weights tensor (e.g., latitude weights). 
                                          Must be broadcastable to the spatial dimensions.
                                          Defaults to None (uniform weighting).
                                          
    Returns:
        torch.Tensor: The ACC per channel, averaged over the batch. Shape: (Channels,)
    """
    # Constraint 1 & 2: All calculations use native PyTorch operations for GPU efficiency.
    
    assert pred.ndim >= 3, (
        f"ACC expects at least 3D tensors (Batch, Channels, ...spatial), got {pred.ndim}D shape {pred.shape}"
    )
    
    # Calculate anomalies
    pred_anomaly = pred - climatology
    target_anomaly = target - climatology

    # If no weights provided, use a weight of 1.0 everywhere
    if weights is None:
        weights = 1.0

    # Determine spatial dimensions dynamically:
    #   4D (Batch, Channels, Lat, Lon)  -> sum over dims (2, 3)
    #   3D (Batch, Channels, Nodes)     -> sum over dim  (2,)
    # This avoids the bug where dim=(-2,-1) on a 3D tensor would
    # sum over Channels AND Nodes, blending all variables together.
    spatial_dims = tuple(range(2, pred.ndim))
    
    # Numerator: sum(weight * pred_anomaly * target_anomaly) over spatial dims
    numerator = torch.sum(weights * pred_anomaly * target_anomaly, dim=spatial_dims)
    
    # Denominator: sqrt(sum(weight * pred_anomaly^2) * sum(weight * target_anomaly^2))
    var_pred = torch.sum(weights * (pred_anomaly ** 2), dim=spatial_dims)
    var_target = torch.sum(weights * (target_anomaly ** 2), dim=spatial_dims)
    
    # Add eps to avoid division by zero
    denominator = torch.sqrt(var_pred * var_target) + 1e-8
    
    # ACC shape after spatial sum: (Batch, Channels)
    acc = numerator / denominator
    
    # Average over the Batch dimension (dim=0) to get ACC per channel
    acc_per_channel = torch.mean(acc, dim=0)

    return acc_per_channel


class ACCMetric(nn.Module):
    """
    nn.Module wrapper for the ACC calculation, allowing seamless integration 
    into PyTorch training/evaluation pipelines.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, climatology: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass to compute ACC.
        """
        return calculate_acc(pred, target, climatology, weights)
