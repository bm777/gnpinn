import torch
import numpy as np
from .base import PDEBase


class Schrodinger1D(PDEBase):
    """
    1D Time-dependent Schrödinger Equation: 
    i * ∂ψ/∂t + (h^2/2m) * ∂^2ψ/∂x^2 - V(x) * ψ = 0
    
    We typically split into real and imaginary parts: ψ = u + iv
    """
    
    def __init__(self, domain_ranges=None, h_bar=1.0, mass=1.0, potential_fn=None, device=None):
        """
        Initialize the 1D Schrödinger Equation.
        
        Args:
            domain_ranges: Dictionary of domain ranges for x and t
            h_bar: Reduced Planck constant
            mass: Particle mass
            potential_fn: Potential function V(x) as callable
            device: Torch device to use
        """
        if domain_ranges is None:
            domain_ranges = {'x': (-5, 5), 't': (0, 1)}
        
        super().__init__(domain_ranges, device)
        self.h_bar = h_bar
        self.mass = mass
        
        # Default potential: harmonic oscillator V(x) = 0.5 * x^2
        if potential_fn is None:
            self.potential_fn = lambda x: 0.5 * x**2
        else:
            self.potential_fn = potential_fn
    
    def compute_residual(self, model, x):
        """
        Compute the Schrödinger equation residual.
        
        Model outputs [ψ_real, ψ_imag] (real and imaginary parts).
        
        Args:
            model: Neural network model
            x: Input points tensor with shape [batch, 2] where x[:, 0] is spatial and x[:, 1] is time
            
        Returns:
            Tensor containing PDE residual values (2 components: real and imaginary parts)
        """
        psi = model(x)
        psi_real = psi[:, 0:1]
        psi_imag = psi[:, 1:2]
        
        # Calculate potential at spatial points
        pot = self.potential_fn(x[:, 0:1])
        
        # First derivatives with respect to t
        grad_t = torch.autograd.grad(
            psi, x,
            grad_outputs=torch.ones_like(psi),
            create_graph=True
        )[0][:, 1:2]
        
        psi_real_t = grad_t[:, 0:1]
        psi_imag_t = grad_t[:, 1:2]
        
        # Second derivatives with respect to x
        grad_x1_real = torch.autograd.grad(
            psi_real, x,
            grad_outputs=torch.ones_like(psi_real),
            create_graph=True
        )[0][:, 0:1]
        
        grad_x1_imag = torch.autograd.grad(
            psi_imag, x,
            grad_outputs=torch.ones_like(psi_imag),
            create_graph=True
        )[0][:, 0:1]
        
        psi_real_xx = torch.autograd.grad(
            grad_x1_real, x,
            grad_outputs=torch.ones_like(grad_x1_real),
            create_graph=True
        )[0][:, 0:1]
        
        psi_imag_xx = torch.autograd.grad(
            grad_x1_imag, x,
            grad_outputs=torch.ones_like(grad_x1_imag),
            create_graph=True
        )[0][:, 0:1]
        
        # Compute the Schrödinger equation residual
        # i * ∂ψ/∂t + (h^2/2m) * ∂^2ψ/∂x^2 - V(x) * ψ = 0
        
        # For real part (from imaginary part of equation):
        # -∂ψ_imag/∂t + (h^2/2m) * ∂^2ψ_real/∂x^2 - V(x) * ψ_real = 0
        h_squared_over_2m = (self.h_bar**2) / (2 * self.mass)
        residual_real = -psi_imag_t + h_squared_over_2m * psi_real_xx - pot * psi_real
        
        # For imaginary part (from real part of equation):
        # ∂ψ_real/∂t + (h^2/2m) * ∂^2ψ_imag/∂x^2 - V(x) * ψ_imag = 0
        residual_imag = psi_real_t + h_squared_over_2m * psi_imag_xx - pot * psi_imag
        
        # Stack residuals
        residual = torch.cat([residual_real, residual_imag], dim=1)
        
        return residual
    
    def get_boundary_conditions(self, n_points=100):
        """
        Define boundary conditions at domain edges.
        Typically, we use zero boundary conditions for Schrödinger.
        
        Returns:
            Dictionary containing boundary points and values
        """
        t = torch.linspace(
            self.domain_ranges['t'][0],
            self.domain_ranges['t'][1],
            n_points,
            device=self.device
        )
        
        # Left boundary
        x_left = torch.ones(n_points, device=self.device) * self.domain_ranges['x'][0]
        bc_left = torch.stack([x_left, t], dim=1)
        
        # Right boundary
        x_right = torch.ones(n_points, device=self.device) * self.domain_ranges['x'][1]
        bc_right = torch.stack([x_right, t], dim=1)
        
        # Combine boundary points
        bc_points = torch.cat([bc_left, bc_right], dim=0)
        
        # Zero boundary values (both real and imaginary parts)
        zeros = torch.zeros(bc_points.shape[0], 1, device=self.device)
        bc_values = torch.cat([zeros, zeros], dim=1)
        
        return {
            'points': bc_points,
            'values': bc_values
        }
    
    def get_initial_conditions(self, n_points=100):
        """
        Define initial condition at t=0, typically a Gaussian wave packet.
        
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
        
        # Initial condition: Gaussian wave packet
        # ψ(x, 0) = exp(-(x-x0)^2/(2*σ^2)) * exp(i*k0*x)
        x0 = 0.0  # center
        sigma = 1.0  # width
        k0 = 1.0  # initial momentum
        
        gaussian = torch.exp(-(x - x0)**2 / (2 * sigma**2))
        
        # Real and imaginary parts of exp(i*k0*x)
        real_part = gaussian * torch.cos(k0 * x)
        imag_part = gaussian * torch.sin(k0 * x)
        
        # Normalize
        norm = torch.sqrt(torch.sum(real_part**2 + imag_part**2)) * (x[1] - x[0])
        real_part = real_part / norm
        imag_part = imag_part / norm
        
        ic_values = torch.stack([real_part, imag_part], dim=1)
        
        return {
            'points': ic_points,
            'values': ic_values
        } 