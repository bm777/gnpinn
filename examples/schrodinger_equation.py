import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MLP, GNPINN
from equations import Schrodinger1D
from utils import set_seed, count_parameters
from visualization import plot_loss_history

# Set random seed for reproducibility
set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define domain
domain_ranges = {
    'x': (-5, 5),  # Spatial domain: x ∈ [-5, 5]
    't': (0, 1),   # Time domain: t ∈ [0, 1]
}

# Create PDE instance with harmonic oscillator potential V(x) = 0.5 * x^2
schrodinger_eq = Schrodinger1D(domain_ranges=domain_ranges, device=device)

# Neural network parameters
input_dim = 2       # (x, t)
output_dim = 2      # (ψ_real, ψ_imag)
hidden_layers = 5
neurons_per_layer = 100

# Create neural network - more neurons for Schrödinger
model = MLP(input_dim, output_dim, hidden_layers, neurons_per_layer)
print(f"Model has {count_parameters(model)} trainable parameters")

# Create GNPINN
pinn = GNPINN(model, schrodinger_eq, device)

# Optimizer
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training parameters
n_epochs = 10000
n_collocation_points = 10000
log_interval = 500

print("Training GN-PINN model for Schrödinger equation...")
history = pinn.train(optimizer, n_collocation_points, n_epochs, log_interval)
print("Training complete!")

# Plot loss history
plot_loss_history(history)
plt.savefig('schrodinger_loss_history.png')

# Plot solution evolution over time
def plot_wavefunction_evolution(model, pde, n_points=100, n_times=5):
    """
    Plot the evolution of the wavefunction (probability density) over time.
    """
    x_min, x_max = pde.domain_ranges['x']
    t_min, t_max = pde.domain_ranges['t']
    
    # Create spatial grid
    x = torch.linspace(x_min, x_max, n_points, device=pde.device)
    
    # Create time points for visualization
    times = np.linspace(t_min, t_max, n_times)
    
    plt.figure(figsize=(12, 10))
    
    for i, t_value in enumerate(times):
        # Create time tensor
        t = torch.ones_like(x, device=pde.device) * t_value
        
        # Create input points
        points = torch.stack([x, t], dim=1)
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            psi = model(points)
            psi_real = psi[:, 0].cpu().numpy()
            psi_imag = psi[:, 1].cpu().numpy()
            
            # Calculate probability density |ψ|^2
            prob_density = psi_real**2 + psi_imag**2
        
        # Plot probability density
        plt.subplot(n_times, 1, i+1)
        plt.plot(x.cpu().numpy(), prob_density)
        plt.title(f'Probability Density at t = {t_value:.2f}')
        plt.xlabel('x')
        plt.ylabel('|ψ|²')
        plt.grid(True)
    
    plt.tight_layout()

# Plot wavefunction evolution
plot_wavefunction_evolution(model, schrodinger_eq)
plt.savefig('schrodinger_evolution.png')

# Create animation of wavefunction evolution
def create_wavefunction_animation(model, pde, filename='schrodinger_animation.mp4', n_points=200, n_frames=100):
    """
    Create an animation of the wavefunction evolution.
    """
    try:
        import matplotlib.animation as animation
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("matplotlib.animation is not available. Animation not created.")
        return
    
    x_min, x_max = pde.domain_ranges['x']
    t_min, t_max = pde.domain_ranges['t']
    
    # Create spatial grid
    x = torch.linspace(x_min, x_max, n_points, device=pde.device)
    x_np = x.cpu().numpy()
    
    # Setup figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    line_real, = ax1.plot([], [], 'b-', label='Real')
    line_imag, = ax1.plot([], [], 'r-', label='Imaginary')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('ψ')
    ax1.set_title('Wavefunction')
    ax1.legend()
    ax1.grid(True)
    
    line_prob, = ax2.plot([], [], 'g-')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(0, 0.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('|ψ|²')
    ax2.set_title('Probability Density')
    ax2.grid(True)
    
    # Time text
    time_text = ax1.text(0.02, 0.9, '', transform=ax1.transAxes)
    
    # Initialization function
    def init():
        line_real.set_data([], [])
        line_imag.set_data([], [])
        line_prob.set_data([], [])
        time_text.set_text('')
        return line_real, line_imag, line_prob, time_text
    
    # Update function for animation
    def update(frame):
        t_value = t_min + frame * (t_max - t_min) / n_frames
        
        # Create time tensor
        t = torch.ones_like(x, device=pde.device) * t_value
        
        # Create input points
        points = torch.stack([x, t], dim=1)
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            psi = model(points)
            psi_real = psi[:, 0].cpu().numpy()
            psi_imag = psi[:, 1].cpu().numpy()
            
            # Calculate probability density |ψ|^2
            prob_density = psi_real**2 + psi_imag**2
        
        line_real.set_data(x_np, psi_real)
        line_imag.set_data(x_np, psi_imag)
        line_prob.set_data(x_np, prob_density)
        time_text.set_text(f't = {t_value:.2f}')
        
        return line_real, line_imag, line_prob, time_text
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True)
    
    # Save animation
    try:
        ani.save(filename, writer='ffmpeg')
        print(f"Animation saved to {filename}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
    
    plt.close()

# Try to create animation (requires ffmpeg)
try:
    create_wavefunction_animation(model, schrodinger_eq)
except Exception as e:
    print(f"Could not create animation: {e}")

print("Plots saved to disk. Example complete.")

# Show plots if running interactively
plt.show() 