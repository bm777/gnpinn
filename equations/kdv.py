import torch
import numpy as np
from .base import PDEBase


class KdV(PDEBase):
    """
    Korteweg-de Vries (KdV) Equation:
    ∂u/∂t + u * ∂u/∂x + β * ∂³u/∂x³ = 0
    
    This nonlinear PDE admits soliton solutions.
    """
    
    def __init__(self, domain_ranges=None, beta=0.0025, device=None):
        """
        Initialize the KdV Equation.
        
        Args:
            domain_ranges: Dictionary of domain ranges for x and t
            beta: Dispersion coefficient (default: 0.0025 for classic KdV)
            device: Torch device to use
        """
        if domain_ranges is None:
            domain_ranges = {'x': (-10, 10), 't': (0, 5)}
        
        super().__init__(domain_ranges, device)
        self.beta = beta
    
    def compute_residual(self, model, x):
        """
        Compute the KdV equation residual:
        ∂u/∂t + u * ∂u/∂x + β * ∂³u/∂x³ = 0
        
        Args:
            model: Neural network model
            x: Input points tensor with shape [batch, 2] where x[:, 0] is x and x[:, 1] is t
            
        Returns:
            Tensor containing PDE residual values
        """
        u = model(x)
        
        # First derivative with respect to t: ∂u/∂t
        u_t = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0][:, 1:2]
        
        # First derivative with respect to x: ∂u/∂x
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0][:, 0:1]
        
        # Second derivative with respect to x: ∂²u/∂x²
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0][:, 0:1]
        
        # Third derivative with respect to x: ∂³u/∂x³
        u_xxx = torch.autograd.grad(
            u_xx, x,
            grad_outputs=torch.ones_like(u_xx),
            create_graph=True
        )[0][:, 0:1]
        
        # KdV residual: ∂u/∂t + u * ∂u/∂x + β * ∂³u/∂x³
        residual = u_t + u * u_x + self.beta * u_xxx
        
        return residual
    
    def get_boundary_conditions(self, n_points=100):
        """
        Define boundary conditions at spatial domain edges.
        
        Returns:
            Dictionary containing boundary points and values
        """
        t = torch.linspace(
            self.domain_ranges['t'][0],
            self.domain_ranges['t'][1],
            n_points,
            device=self.device
        )
        
        # Left boundary (x = x_min)
        x_left = torch.ones(n_points, device=self.device) * self.domain_ranges['x'][0]
        bc_left = torch.stack([x_left, t], dim=1)
        
        # Right boundary (x = x_max)
        x_right = torch.ones(n_points, device=self.device) * self.domain_ranges['x'][1]
        bc_right = torch.stack([x_right, t], dim=1)
        
        # Combined boundary points
        bc_points = torch.cat([bc_left, bc_right], dim=0)
        
        # Zero boundary values for KdV (assuming soliton solution with zero at boundaries)
        bc_values = torch.zeros(bc_points.shape[0], 1, device=self.device)
        
        return {
            'points': bc_points,
            'values': bc_values
        }
    
    def get_initial_conditions(self, n_points=200, amplitude=1.0, width=1.0, x0=0.0):
        """
        Define initial condition at t=0, typically a soliton.
        
        Args:
            n_points: Number of points
            amplitude: Initial soliton amplitude
            width: Soliton width parameter
            x0: Initial soliton position
            
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
        
        # Initial condition: soliton u(x, 0) = A * sech²((x-x0)/w)
        ic_values = amplitude * torch.pow(1.0 / torch.cosh((x - x0) / width), 2).unsqueeze(1)
        
        return {
            'points': ic_points,
            'values': ic_values
        }
    
    def exact_solution(self, x, t, amplitude=1.0, width=1.0, x0=0.0):
        """
        Compute exact soliton solution for KdV.
        
        The classic single-soliton solution is:
        u(x, t) = A * sech²((x - x0 - ct)/w)
        where c = A/3 is the soliton speed.
        
        Args:
            x: Spatial coordinates tensor
            t: Time coordinates tensor
            amplitude: Soliton amplitude
            width: Soliton width parameter
            x0: Initial soliton position
            
        Returns:
            Exact solution tensor
        """
        try:
            # Convert numpy arrays to torch tensors if needed
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, device=self.device)
            if isinstance(t, np.ndarray):
                t = torch.tensor(t, device=self.device)
            
            # Soliton speed depends on amplitude
            c = amplitude / 3.0
            
            # Single soliton solution
            xi = x - x0 - c * t  # Moving coordinate
            u = amplitude * torch.pow(1.0 / torch.cosh(xi / width), 2)
            
            return u
        except RuntimeError as e:
            print(f"Warning in exact solution: {e}")
            # Try with numpy fallback if torch fails
            if not isinstance(x, torch.Tensor):
                c = amplitude / 3.0
                xi = x - x0 - c * t
                return amplitude * np.power(1.0 / np.cosh(xi / width), 2)
            else:
                # Use CPU tensors if device conversion fails
                c = amplitude / 3.0
                xi = x.cpu() - x0 - c * t.cpu()
                return amplitude * torch.pow(1.0 / torch.cosh(xi / width), 2) 