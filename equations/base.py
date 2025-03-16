import torch
import abc


class PDEBase(abc.ABC):
    """
    Abstract base class for PDE equations.
    All specific PDE implementations should inherit from this class.
    """
    
    def __init__(self, domain_ranges, device=None):
        """
        Initialize the PDE.
        
        Args:
            domain_ranges: Dictionary of domain ranges for each dimension
            device: Torch device to use (cpu or cuda)
        """
        self.domain_ranges = domain_ranges
        self.device = torch.device("cpu") if device is None else device
    
    @abc.abstractmethod
    def compute_residual(self, model, x):
        """
        Compute the PDE residual at given points.
        
        Args:
            model: Neural network model
            x: Input points tensor
        
        Returns:
            Tensor containing PDE residual values
        """
        pass
    
    @abc.abstractmethod
    def get_boundary_conditions(self):
        """
        Define the boundary conditions for this PDE.
        
        Returns:
            Dictionary containing boundary points and values
        """
        pass
    
    @abc.abstractmethod
    def get_initial_conditions(self):
        """
        Define the initial conditions for this PDE.
        
        Returns:
            Dictionary containing initial points and values
        """
        pass
    
    def generate_collocation_points(self, num_points):
        """
        Generate collocation points for PDE residual evaluation.
        
        Args:
            num_points: Number of collocation points to generate
            
        Returns:
            Tensor of collocation points
        """
        # Default implementation for rectangular domains
        points = []
        for dim, (lower, upper) in self.domain_ranges.items():
            points_dim = lower + (upper - lower) * torch.rand(num_points, 1, device=self.device)
            points.append(points_dim)
        
        collocation_points = torch.cat(points, dim=1)
        collocation_points.requires_grad_(True)
        return collocation_points 