import torch
import torch.optim as optim
import torch.nn as nn # For activation function like nn.Tanh()

# Assuming models and equations are in the path or PYTHONPATH is set correctly
# For a typical project structure, you might use relative imports if this script
# is part of a larger package, but for a standalone example, direct imports
# might require the top-level directory to be in PYTHONPATH.
# For now, let's assume they can be imported directly.
from models.neural_networks import MLP # Using the existing MLP class
from equations.navier_stokes import NavierStokes3D
from models.pinn import GNPINN

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Problem Definition
    domain_ranges = {'x': [0, 1], 'y': [0, 1], 'z': [0, 1], 't': [0, 1]}
    viscosity = 0.01
    density = 1.0
    print(f"Problem Definition: Domain={domain_ranges}, Viscosity={viscosity}, Density={density}")

    # 3. Instantiate PDE
    pde = NavierStokes3D(domain_ranges=domain_ranges, 
                         viscosity=viscosity, 
                         density=density, 
                         device=device)
    print("NavierStokes3D PDE instantiated.")

    # 4. Instantiate Neural Network Model
    input_dim = 4  # x, y, z, t
    output_dim = 4 # u, v, w, p
    hidden_layers = 3
    neurons_per_layer = 64 # Corresponds to [64, 64, 64] in SimpleMLP
    activation_fn = nn.Tanh() # Using nn.Tanh() directly
    
    # The existing MLP takes string or callable for activation.
    # If passing a callable like nn.Tanh(), it should work.
    # Let's ensure the MLP class can handle it.
    # From reading the MLP class, it seems it expects a string, then converts.
    # Let's pass 'tanh' as a string.
    model = MLP(input_dim=input_dim, 
                output_dim=output_dim, 
                hidden_layers=hidden_layers, 
                neurons_per_layer=neurons_per_layer,
                activation='tanh') # Using string 'tanh' as expected by MLP
    model.to(device)
    print(f"Neural Network Model (MLP) instantiated with {hidden_layers} hidden layers and {neurons_per_layer} neurons/layer.")
    print(model)

    # 5. Instantiate GNPINN Solver
    gnpinn = GNPINN(model=model, pde=pde, device=device)
    print("GNPINN solver instantiated.")

    # 6. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"Optimizer: Adam with lr={1e-3}")

    # 7. Training
    n_collocation_points = 20000 # As suggested
    n_boundary_points = 500 # For each boundary type (initial, 6 faces) - pde.get_boundary_conditions will handle specific distribution
    n_initial_points = 1000  # For initial conditions
    n_epochs = 5000 # As suggested
    log_interval = 100 # As suggested

    print(f"\nStarting training for {n_epochs} epochs...")
    print(f"Collocation points: {n_collocation_points}")
    print(f"Boundary points parameter for GNPINN: {n_boundary_points}")
    print(f"Initial points parameter for GNPINN: {n_initial_points}")

    # Note: The GNPINN.train method in the problem description took:
    # optimizer, n_collocation_points, n_epochs, log_interval
    # I'll assume it internally calls pde.get_boundary_conditions and pde.get_initial_conditions
    # and uses n_boundary_points and n_initial_points for those.
    # If the GNPINN.train signature is different, this might need adjustment.
    # For now, I'll pass the n_boundary_points and n_initial_points to train,
    # assuming the GNPINN class knows how to use them.
    
    # Let's check the GNPINN train signature from its definition if possible.
    # Assuming it's: train(self, optimizer, n_epochs, n_collocation, n_boundary=None, n_initial=None, log_interval=100, batch_size=None)
    # The problem description for GNPINN.train was: gnpinn.train(...) with the optimizer, collocation points, epochs, and a reasonable log_interval
    # It did not specify n_boundary_points or n_initial_points for the train call.
    # I will stick to the provided signature and assume GNPINN handles BC/IC point generation internally using defaults or fixed numbers.
    # The PDE methods get_boundary_conditions and get_initial_conditions have n_points parameters.
    # Let's assume GNPINN calls these with some default or uses a fixed strategy.
    # For a more robust example, one might expose these numbers in the train call.
    # Given the prompt, I will use the simpler train call.

    gnpinn.train(optimizer=optimizer,
                 n_epochs=n_epochs,
                 n_collocation=n_collocation_points,
                 # n_boundary=n_boundary_points, # Assuming GNPINN handles this
                 # n_initial=n_initial_points,   # Assuming GNPINN handles this
                 log_interval=log_interval)

    print("\nTraining finished.")

    # Example: Evaluate model at some points (optional)
    # test_points = torch.rand((10, input_dim), device=device)
    # test_points[:, 0] = test_points[:, 0] * (domain_ranges['x'][1] - domain_ranges['x'][0]) + domain_ranges['x'][0]
    # test_points[:, 1] = test_points[:, 1] * (domain_ranges['y'][1] - domain_ranges['y'][0]) + domain_ranges['y'][0]
    # test_points[:, 2] = test_points[:, 2] * (domain_ranges['z'][1] - domain_ranges['z'][0]) + domain_ranges['z'][0]
    # test_points[:, 3] = test_points[:, 3] * (domain_ranges['t'][1] - domain_ranges['t'][0]) + domain_ranges['t'][0]
    # model.eval()
    # with torch.no_grad():
    #    predictions = model(test_points)
    # print("\nExample predictions (u, v, w, p) for 10 random points:")
    # for i in range(predictions.shape[0]):
    #    print(f"Point ({test_points[i,0]:.2f}, {test_points[i,1]:.2f}, {test_points[i,2]:.2f}, {test_points[i,3]:.2f}) -> Pred ({predictions[i,0]:.4f}, {predictions[i,1]:.4f}, {predictions[i,2]:.4f}, {predictions[i,3]:.4f})")

if __name__ == "__main__":
    main()
