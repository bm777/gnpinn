import torch
import numpy as np
from .base import PDEBase


class NavierStokes2D(PDEBase):
    """
    2D incompressible Navier-Stokes equations for crystal growth problems.
    
    Equations:
    ∂u/∂t + u·∇u = -(1/ρ)∇p + ν∇²u
    ∇·u = 0
    
    where:
    - u(x,y,t) is the velocity field (u, v components)
    - p(x,y,t) is pressure
    - ρ is density
    - ν is kinematic viscosity
    
    For crystal growth, we also couple this with temperature equation:
    ∂T/∂t + u·∇T = α∇²T
    where α is thermal diffusivity.
    """
    
    def __init__(self, domain_ranges=None, viscosity=0.01, thermal_diffusivity=0.005, 
                 density=1.0, crystal_interface_fn=None, device=None):
        """
        Initialize the Navier-Stokes Equations for crystal growth.
        
        Args:
            domain_ranges: Dictionary of domain ranges for x, y and t
            viscosity: Kinematic viscosity coefficient
            thermal_diffusivity: Thermal diffusivity
            density: Fluid density
            crystal_interface_fn: Function defining crystal interface position (if None, uses default)
            device: Torch device to use
        """
        if domain_ranges is None:
            domain_ranges = {'x': (0, 1), 'y': (0, 1), 't': (0, 1)}
        
        super().__init__(domain_ranges, device)
        self.viscosity = viscosity
        self.thermal_diffusivity = thermal_diffusivity
        self.density = density
        
        # Default crystal interface: Flat interface at y=0.2
        if crystal_interface_fn is None:
            self.crystal_interface = lambda x, t: 0.2 * torch.ones_like(x)
        else:
            self.crystal_interface = crystal_interface_fn
    
    def compute_residual(self, model, points):
        """
        Compute the Navier-Stokes and heat equation residuals.
        
        Args:
            model: Neural network model
            points: Input points tensor with shape [batch, 3] where points[:, 0] is x, 
                   points[:, 1] is y, and points[:, 2] is t
            
        Returns:
            Tensor containing PDE residual values (4 components: u, v, p, T)
        """
        # Extract coordinates for easier access
        x = points[:, 0:1]
        y = points[:, 1:2]
        t = points[:, 2:3]
        
        # Get model predictions (u, v, p, T)
        outputs = model(points)
        u = outputs[:, 0:1]  # x-velocity
        v = outputs[:, 1:2]  # y-velocity
        p = outputs[:, 2:3]  # pressure
        T = outputs[:, 3:4]  # temperature
        
        # First derivatives
        grad_u = torch.autograd.grad(
            u, points, grad_outputs=torch.ones_like(u),
            create_graph=True, allow_unused=True
        )[0]
        grad_v = torch.autograd.grad(
            v, points, grad_outputs=torch.ones_like(v),
            create_graph=True, allow_unused=True
        )[0]
        grad_p = torch.autograd.grad(
            p, points, grad_outputs=torch.ones_like(p),
            create_graph=True, allow_unused=True
        )[0]
        grad_T = torch.autograd.grad(
            T, points, grad_outputs=torch.ones_like(T),
            create_graph=True, allow_unused=True
        )[0]
        
        # Extract velocity derivatives
        u_t = grad_u[:, 2:3]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        
        v_t = grad_v[:, 2:3]
        v_x = grad_v[:, 0:1]
        v_y = grad_v[:, 1:2]
        
        # Pressure derivatives
        p_x = grad_p[:, 0:1]
        p_y = grad_p[:, 1:2]
        
        # Temperature derivatives
        T_t = grad_T[:, 2:3]
        T_x = grad_T[:, 0:1]
        T_y = grad_T[:, 1:2]
        
        # Second derivatives for velocity (for viscous terms)
        u_xx = torch.autograd.grad(
            u_x, points, grad_outputs=torch.ones_like(u_x),
            create_graph=True, allow_unused=True
        )[0][:, 0:1]
        
        u_yy = torch.autograd.grad(
            u_y, points, grad_outputs=torch.ones_like(u_y),
            create_graph=True, allow_unused=True
        )[0][:, 1:2]
        
        v_xx = torch.autograd.grad(
            v_x, points, grad_outputs=torch.ones_like(v_x),
            create_graph=True, allow_unused=True
        )[0][:, 0:1]
        
        v_yy = torch.autograd.grad(
            v_y, points, grad_outputs=torch.ones_like(v_y),
            create_graph=True, allow_unused=True
        )[0][:, 1:2]
        
        # Second derivatives for temperature (for diffusion term)
        T_xx = torch.autograd.grad(
            T_x, points, grad_outputs=torch.ones_like(T_x),
            create_graph=True, allow_unused=True
        )[0][:, 0:1]
        
        T_yy = torch.autograd.grad(
            T_y, points, grad_outputs=torch.ones_like(T_y),
            create_graph=True, allow_unused=True
        )[0][:, 1:2]
        
        # Compute PDE residuals
        
        # Momentum equation in x-direction
        fx = u_t + u * u_x + v * u_y + (1/self.density) * p_x - self.viscosity * (u_xx + u_yy)
        
        # Momentum equation in y-direction
        fy = v_t + u * v_x + v * v_y + (1/self.density) * p_y - self.viscosity * (v_xx + v_yy)
        
        # Continuity equation (incompressibility constraint)
        continuity = u_x + v_y
        
        # Temperature equation
        temperature = T_t + u * T_x + v * T_y - self.thermal_diffusivity * (T_xx + T_yy)
        
        # Stack all residuals
        residual = torch.cat([fx, fy, continuity, temperature], dim=1)
        
        return residual
    
    def get_boundary_conditions(self, n_points=100):
        """
        Define boundary conditions for Navier-Stokes with crystal growth.
        
        For crystal growth setting:
        - No-slip boundary at walls (u=v=0)
        - Fixed temperature at top (cooling) and bottom (heating)
        - Crystal interface has special conditions
        
        Returns:
            Dictionary containing boundary points and values
        """
        device = self.device
        
        # Extract domain boundaries
        x_min, x_max = self.domain_ranges['x']
        y_min, y_max = self.domain_ranges['y']
        t_min, t_max = self.domain_ranges['t']
        
        # Time points
        t = torch.linspace(t_min, t_max, n_points, device=device)
        
        # Spatial coordinates for boundary
        x = torch.linspace(x_min, x_max, n_points, device=device)
        y = torch.linspace(y_min, y_max, n_points, device=device)
        
        # Get the dtype from the linspace tensors
        dtype = x.dtype
        
        # Create meshgrids for each boundary - ensure same dtype
        X_bottom = torch.meshgrid(x, torch.tensor([y_min], dtype=dtype, device=device), indexing='ij')[0]
        X_top = torch.meshgrid(x, torch.tensor([y_max], dtype=dtype, device=device), indexing='ij')[0]
        Y_left = torch.meshgrid(torch.tensor([x_min], dtype=dtype, device=device), y, indexing='ij')[1]
        Y_right = torch.meshgrid(torch.tensor([x_max], dtype=dtype, device=device), y, indexing='ij')[1]
        
        # Flatten arrays for each boundary
        x_bottom = X_bottom.flatten()
        y_bottom = torch.ones_like(x_bottom) * y_min
        
        x_top = X_top.flatten()
        y_top = torch.ones_like(x_top) * y_max
        
        x_left = torch.ones_like(Y_left.flatten()) * x_min
        y_left = Y_left.flatten()
        
        x_right = torch.ones_like(Y_right.flatten()) * x_max
        y_right = Y_right.flatten()
        
        # Create boundary points for all time steps
        bc_points = []
        bc_values = []
        
        for current_t in t:
            # Bottom boundary (heated wall/crystal interface)
            t_bottom = torch.ones_like(x_bottom) * current_t
            boundary_bottom = torch.stack([x_bottom, y_bottom, t_bottom], dim=1)
            
            # Values at bottom: no slip (u=v=0), high temperature
            u_bottom = torch.zeros_like(x_bottom)
            v_bottom = torch.zeros_like(x_bottom)
            p_bottom = torch.zeros_like(x_bottom)  # Pressure set to zero at boundary
            T_bottom = torch.ones_like(x_bottom)   # Hot temperature (normalized to 1)
            
            values_bottom = torch.stack([u_bottom, v_bottom, p_bottom, T_bottom], dim=1)
            
            # Top boundary (cooling wall)
            t_top = torch.ones_like(x_top) * current_t
            boundary_top = torch.stack([x_top, y_top, t_top], dim=1)
            
            # Values at top: no slip, low temperature
            u_top = torch.zeros_like(x_top)
            v_top = torch.zeros_like(x_top)
            p_top = torch.zeros_like(x_top)
            T_top = torch.zeros_like(x_top)  # Cold temperature (normalized to 0)
            
            values_top = torch.stack([u_top, v_top, p_top, T_top], dim=1)
            
            # Left boundary (wall)
            t_left = torch.ones_like(y_left) * current_t
            boundary_left = torch.stack([x_left, y_left, t_left], dim=1)
            
            # Values at left: no slip, insulated
            u_left = torch.zeros_like(y_left)
            v_left = torch.zeros_like(y_left)
            p_left = torch.zeros_like(y_left)
            
            # Linear temperature gradient between hot and cold
            T_left = 1.0 - (y_left - y_min) / (y_max - y_min)
            
            values_left = torch.stack([u_left, v_left, p_left, T_left], dim=1)
            
            # Right boundary (wall)
            t_right = torch.ones_like(y_right) * current_t
            boundary_right = torch.stack([x_right, y_right, t_right], dim=1)
            
            # Values at right: no slip, insulated
            u_right = torch.zeros_like(y_right)
            v_right = torch.zeros_like(y_right)
            p_right = torch.zeros_like(y_right)
            
            # Linear temperature gradient between hot and cold
            T_right = 1.0 - (y_right - y_min) / (y_max - y_min)
            
            values_right = torch.stack([u_right, v_right, p_right, T_right], dim=1)
            
            # Combine all boundaries
            bc_points.append(torch.cat([boundary_bottom, boundary_top, boundary_left, boundary_right], dim=0))
            bc_values.append(torch.cat([values_bottom, values_top, values_left, values_right], dim=0))
        
        # Stack over time dimension
        bc_points = torch.cat(bc_points, dim=0)
        bc_values = torch.cat(bc_values, dim=0)
        
        return {
            'points': bc_points,
            'values': bc_values
        }
    
    def get_initial_conditions(self, n_points_x=50, n_points_y=50):
        """
        Define initial conditions for Navier-Stokes with crystal growth.
        
        Starting with fluid at rest with a temperature gradient.
        
        Returns:
            Dictionary containing initial points and values
        """
        device = self.device
        
        # Extract domain boundaries
        x_min, x_max = self.domain_ranges['x']
        y_min, y_max = self.domain_ranges['y']
        
        # Create spatial grid
        x = torch.linspace(x_min, x_max, n_points_x, device=device)
        y = torch.linspace(y_min, y_max, n_points_y, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        X_flat = X.flatten().unsqueeze(1)
        Y_flat = Y.flatten().unsqueeze(1)
        T_flat = torch.zeros_like(X_flat)  # t=0
        
        # Create initial points at t=0
        ic_points = torch.cat([X_flat, Y_flat, T_flat], dim=1)
        
        # Initial conditions: fluid at rest (u=v=0)
        u_init = torch.zeros_like(X_flat)
        v_init = torch.zeros_like(X_flat)
        p_init = torch.zeros_like(X_flat)
        
        # Initial temperature: Linear gradient from bottom (hot) to top (cold)
        T_init = 1.0 - (Y_flat - y_min) / (y_max - y_min)
        
        # Stack all values
        ic_values = torch.cat([u_init, v_init, p_init, T_init], dim=1)
        
        return {
            'points': ic_points,
            'values': ic_values
        }
    
    def get_crystal_interface_points(self, t, n_points=50):
        """
        Generate points along the crystal interface for a given time.
        
        Args:
            t: Time value
            n_points: Number of points along interface
            
        Returns:
            Dictionary with points and values at the crystal interface
        """
        device = self.device
        
        # Extract x domain
        x_min, x_max = self.domain_ranges['x']
        
        # Create x coordinates along interface
        x = torch.linspace(x_min, x_max, n_points, device=device)
        
        # Get interface y-position (can be time-dependent for growing crystal)
        y = self.crystal_interface(x, t)
        
        # Create time array
        t_tensor = torch.ones_like(x) * t
        
        # Stack to get points
        interface_points = torch.stack([x, y, t_tensor], dim=1)
        
        # Values at interface:
        # - No slip (u=v=0)
        # - Fixed temperature (T=1, hot)
        u_interface = torch.zeros_like(x)
        v_interface = torch.zeros_like(x)
        p_interface = torch.zeros_like(x)
        T_interface = torch.ones_like(x)
        
        interface_values = torch.stack([u_interface, v_interface, p_interface, T_interface], dim=1)
        
        return {
            'points': interface_points,
            'values': interface_values
        }
    
    def generate_collocation_points(self, num_points):
        """
        Generate collocation points excluding the crystal region.
        
        Args:
            num_points: Number of collocation points to generate
            
        Returns:
            Tensor of collocation points
        """
        # Start with uniform random points
        points = super().generate_collocation_points(num_points)
        
        # Extract coordinates
        x = points[:, 0]
        y = points[:, 1]
        t = points[:, 2]
        
        # Get crystal interface position
        interface_y = self.crystal_interface(x, t)
        
        # Create mask for points above the crystal (in fluid region)
        fluid_mask = y > interface_y
        
        # Filter points to only include fluid region
        filtered_points = points[fluid_mask]
        
        # If we filtered out too many points, generate more
        if filtered_points.shape[0] < num_points / 2:
            extra_points = super().generate_collocation_points(num_points)
            
            # Extract coordinates
            x_extra = extra_points[:, 0]
            y_extra = extra_points[:, 1]
            t_extra = extra_points[:, 2]
            
            # Get crystal interface position
            interface_y_extra = self.crystal_interface(x_extra, t_extra)
            
            # Create mask for points above the crystal (in fluid region)
            fluid_mask_extra = y_extra > interface_y_extra
            
            # Filter additional points
            filtered_extra_points = extra_points[fluid_mask_extra]
            
            # Combine with original filtered points
            filtered_points = torch.cat([filtered_points, filtered_extra_points], dim=0)
            
            # Trim to desired number if necessary
            if filtered_points.shape[0] > num_points:
                filtered_points = filtered_points[:num_points]
        
        # Ensure requires_grad is set
        filtered_points.requires_grad_(True)
        
        return filtered_points 