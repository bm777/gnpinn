import torch
import numpy as np
import os
import json


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path, metadata=None):
    """
    Save model weights and metadata.
    
    Args:
        model: PyTorch model
        path: Path to save the model
        metadata: Additional metadata to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), path)
    
    # Save metadata if provided
    if metadata:
        metadata_path = os.path.splitext(path)[0] + '.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)


def load_model(model, path):
    """
    Load model weights.
    
    Args:
        model: PyTorch model
        path: Path to the saved model
        
    Returns:
        Loaded model and metadata (if available)
    """
    model.load_state_dict(torch.load(path))
    
    # Try to load metadata if available
    metadata = None
    metadata_path = os.path.splitext(path)[0] + '.json'
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata


def exact_solution_heat_1d(x, t, diffusivity=1.0):
    """
    Exact solution for 1D heat equation with initial condition u(x, 0) = sin(πx)
    and boundary conditions u(0, t) = u(1, t) = 0.
    
    Args:
        x: Spatial coordinate array
        t: Time coordinate array
        diffusivity: Heat diffusion coefficient
        
    Returns:
        Solution array
    """
    try:
        # Try with numpy if x is numpy array
        if isinstance(x, np.ndarray):
            return np.sin(np.pi * x) * np.exp(-diffusivity * (np.pi**2) * t)
        # Handle torch tensor input
        elif isinstance(x, torch.Tensor):
            return torch.sin(torch.pi * x) * torch.exp(-diffusivity * (torch.pi**2) * t)
        # Convert list to numpy array if needed
        else:
            return np.sin(np.pi * np.array(x)) * np.exp(-diffusivity * (np.pi**2) * np.array(t))
    except RuntimeError:
        # Fallback to torch tensors if numpy is not available
        x_tensor = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        return torch.sin(torch.pi * x_tensor) * torch.exp(-diffusivity * (torch.pi**2) * t_tensor) 