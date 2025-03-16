import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MLP, GNPINN
from equations import Heat1D
from utils import set_seed, count_parameters, exact_solution_heat_1d
from visualization import plot_loss_history, plot_solution_1d, plot_solution_comparison_1d

# Set random seed for reproducibility
set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# PDE parameters
diffusivity = 1.0

# Define domain
domain_ranges = {
    'x': (0, 1),  # Spatial domain: x ∈ [0, 1]
    't': (0, 1),  # Time domain: t ∈ [0, 1]
}

# Create PDE instance
heat_eq = Heat1D(domain_ranges=domain_ranges, diffusivity=diffusivity, device=device)

# Neural network parameters
input_dim = 2  # (x, t)
output_dim = 1  # u(x, t)
hidden_layers = 4
neurons_per_layer = 50

# Create neural network
model = MLP(input_dim, output_dim, hidden_layers, neurons_per_layer)
print(f"Model has {count_parameters(model)} trainable parameters")

# Create GNPINN
pinn = GNPINN(model, heat_eq, device)

# Optimizer
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training parameters
n_epochs = 5000
n_collocation_points = 10000
log_interval = 100

print("Training GN-PINN model...")
history = pinn.train(optimizer, n_collocation_points, n_epochs, log_interval)
print("Training complete!")

# Plot loss history
plot_loss_history(history)
plt.savefig('heat_loss_history.png')

# Plot solution
plot_solution_1d(model, heat_eq)
plt.savefig('heat_solution.png')

# Compare with exact solution
def exact_solution_wrapper(x, t):
    return exact_solution_heat_1d(x, t, diffusivity)

plot_solution_comparison_1d(model, heat_eq, exact_solution_wrapper)
plt.savefig('heat_solution_comparison.png')

print("Plots saved to disk. Example complete.")

# Show plots if running interactively
plt.show() 