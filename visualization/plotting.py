import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_loss_history(history, log_scale=True, figsize=(12, 5)):
    """
    Plot the loss history from training.
    
    Args:
        history: Dictionary containing loss values
        log_scale: Whether to use log scale for y-axis
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot PDE loss
    plt.subplot(1, 3, 1)
    plt.plot(history['pde_loss'])
    plt.title('PDE Loss')
    plt.xlabel('Epoch')
    if log_scale:
        plt.yscale('log')
    
    # Plot BC/IC loss
    plt.subplot(1, 3, 2)
    plt.plot(history['bc_ic_loss'])
    plt.title('BC/IC Loss')
    plt.xlabel('Epoch')
    if log_scale:
        plt.yscale('log')
    
    # Plot total loss
    plt.subplot(1, 3, 3)
    plt.plot(history['total_loss'])
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    if log_scale:
        plt.yscale('log')
    
    plt.tight_layout()
    
    
def plot_solution_1d(model, pde, n_points=100, figsize=(12, 5)):
    """
    Plot the 1D PDE solution.
    
    Args:
        model: Trained neural network model
        pde: PDE object
        n_points: Number of points to evaluate at
        figsize: Figure size
    """
    # Extract domain ranges from PDE
    x_min, x_max = pde.domain_ranges['x']
    t_min, t_max = pde.domain_ranges['t']
    
    # Create meshgrid for evaluation
    x = torch.linspace(x_min, x_max, n_points, device=pde.device)
    t = torch.linspace(t_min, t_max, n_points, device=pde.device)
    X, T = torch.meshgrid(x, t, indexing='ij')
    
    X_flat = X.flatten().unsqueeze(1)
    T_flat = T.flatten().unsqueeze(1)
    
    points = torch.cat([X_flat, T_flat], dim=1)
    
    # Evaluate model
    model.eval()
    try:
        with torch.no_grad():
            u_pred = model(points).cpu().numpy().reshape(n_points, n_points)
            X_np, T_np = X.cpu().numpy(), T.cpu().numpy()
    except RuntimeError as e:
        print(f"Warning: Error converting to NumPy: {e}")
        print("Using PyTorch tensors directly for plotting...")
        with torch.no_grad():
            u_pred = model(points).cpu().reshape(n_points, n_points)
            X_np, T_np = X.cpu(), T.cpu()
    
    # Create plots
    plt.figure(figsize=figsize)
    
    try:
        # Surface plot
        ax1 = plt.subplot(1, 2, 1, projection='3d')
        surf = ax1.plot_surface(X_np, T_np, u_pred, cmap='viridis', edgecolor='none')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_zlabel('u(x,t)')
        ax1.set_title('Solution Surface')
        plt.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
        
        # Contour plot
        ax2 = plt.subplot(1, 2, 2)
        contour = ax2.contourf(X_np, T_np, u_pred, cmap='viridis', levels=20)
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_title('Solution Contour')
        plt.colorbar(contour, ax=ax2, shrink=0.5, aspect=10)
    except Exception as e:
        print(f"Warning: Error in plotting: {e}")
        print("Attempting simplified plot...")
        plt.clf()  # Clear the current figure
        
        # Simplified 2D plots at different time slices
        t_slices = [0, n_points//4, n_points//2, 3*n_points//4, n_points-1]
        for i, t_idx in enumerate(t_slices):
            plt.subplot(len(t_slices), 1, i+1)
            if isinstance(u_pred, torch.Tensor):
                plt.plot(X_np[:, 0].tolist(), u_pred[:, t_idx].tolist())
            else:
                plt.plot(X_np[:, 0], u_pred[:, t_idx])
            plt.title(f't = {T_np[0, t_idx]:.2f}')
            plt.grid(True)
    
    plt.tight_layout()
    

def plot_solution_comparison_1d(model, pde, exact_solution_fn, t_samples=[0.0, 0.25, 0.5, 0.75, 1.0], n_points=100, figsize=(12, 10)):
    """
    Compare PINN solution with exact solution at specific time slices.
    
    Args:
        model: Trained neural network model
        pde: PDE object
        exact_solution_fn: Function that computes exact solution given x, t
        t_samples: List of time points to plot
        n_points: Number of points to evaluate at
        figsize: Figure size
    """
    # Extract spatial domain from PDE
    x_min, x_max = pde.domain_ranges['x']
    
    # Create spatial grid
    x = torch.linspace(x_min, x_max, n_points, device=pde.device)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    for i, t_value in enumerate(t_samples):
        # Create time tensor
        t = torch.ones_like(x, device=pde.device) * t_value
        
        # Create input points
        points = torch.stack([x, t], dim=1)
        
        # Evaluate model
        model.eval()
        try:
            with torch.no_grad():
                u_pred = model(points).cpu().numpy().flatten()
                x_np = x.cpu().numpy()
                t_np = t.cpu().numpy()
                u_exact = exact_solution_fn(x_np, t_np)
        except RuntimeError as e:
            print(f"Warning: Error converting to NumPy: {e}")
            print("Using PyTorch tensors directly for plotting...")
            with torch.no_grad():
                u_pred = model(points).cpu().flatten()
                x_np = x.cpu()
                t_np = t.cpu()
                # Try to compute exact solution with PyTorch tensors
                try:
                    u_exact = exact_solution_fn(x_np, t_np)
                except:
                    print("Warning: Could not compute exact solution with PyTorch tensors")
                    u_exact = torch.zeros_like(u_pred)  # Placeholder
        
        # Plot comparison
        plt.subplot(len(t_samples), 1, i+1)
        if isinstance(u_pred, torch.Tensor) and isinstance(x_np, torch.Tensor):
            plt.plot(x_np.tolist(), u_pred.tolist(), 'r-', label='PINN')
            plt.plot(x_np.tolist(), u_exact.tolist() if isinstance(u_exact, torch.Tensor) else u_exact, 'b--', label='Exact')
        else:
            plt.plot(x_np, u_pred, 'r-', label='PINN')
            plt.plot(x_np, u_exact, 'b--', label='Exact')
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.title(f't = {t_value}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout() 