import torch
import numpy as np
from .base import PDEBase


class Heat1D(PDEBase):
    """
    1D Heat Equation: du/dt - diffusivity * d^2u/dx^2 = 0
    """
    
    def __init__(self, domain_ranges=None, diffusivity=1.0, device=None):
        """
        Initialize the 1D Heat Equation.
        
        Args:
            domain_ranges: Dictionary of domain ranges for x and t
            diffusivity: Heat diffusion coefficient
            device: Torch device to use
        """
        if domain_ranges is None:
            domain_ranges = {'x': (0, 1), 't': (0, 1)}
        
        super().__init__(domain_ranges, device)
        self.diffusivity = diffusivity
        
    def compute_residual(self, model, x):
        """
        Compute the heat equation residual: du/dt - diffusivity * d^2u/dx^2
        
        Args:
            model: Neural network model
            x: Input points tensor with shape [batch, 2] where x[:, 0] is spatial dim and x[:, 1] is time
        
        Returns:
            Tensor containing PDE residual values
        """
        u = model(x)
        
        # Compute du/dt
        u_t = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0][:, 1:2]
        
        # Compute d^2u/dx^2
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0][:, 0:1]
        
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0][:, 0:1]
        
        # Heat equation residual
        residual = u_t - self.diffusivity * u_xx
        
        return residual
    
    def get_boundary_conditions(self, n_points=100):
        """
        Define Dirichlet boundary conditions at x=0 and x=1.
        
        Returns:
            Dictionary containing boundary points and values
        """
        t = torch.linspace(
            self.domain_ranges['t'][0],
            self.domain_ranges['t'][1],
            n_points,
            device=self.device
        )
        
        # x=0 boundary
        x_left = torch.zeros_like(t, device=self.device)
        bc_left = torch.stack([x_left, t], dim=1)
        
        # x=1 boundary
        x_right = torch.ones_like(t, device=self.device)
        bc_right = torch.stack([x_right, t], dim=1)
        
        # Combine boundary points
        bc_points = torch.cat([bc_left, bc_right], dim=0)
        
        # Boundary values (zero Dirichlet boundary for this example)
        bc_values = torch.zeros(bc_points.shape[0], 1, device=self.device)
        
        return {
            'points': bc_points,
            'values': bc_values
        }
    
    def get_initial_conditions(self, n_points=100):
        """
        Define initial condition at t=0.
        
        Returns:
            Dictionary containing initial points and values
        """
        x = torch.linspace(
            self.domain_ranges['x'][0],
            self.domain_ranges['x'][1],
            n_points,
            device=self.device
        )
        
        t_zero = torch.zeros_like(x, device=self.device)
        ic_points = torch.stack([x, t_zero], dim=1)
        
        # Initial condition: u(x, 0) = sin(π*x)
        ic_values = torch.sin(np.pi * x).unsqueeze(1)
        
        return {
            'points': ic_points,
            'values': ic_values
        } 