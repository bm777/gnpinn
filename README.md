# GNPINN: Gradient-Normalized Physics-Informed Neural Networks

A modular framework implementing Physics-Informed Neural Networks (PINNs) with Gradient Normalization for solving differential equations including Navier-Stokes equations for crystal growth modeling.

## Features

- **Modular Design**: Easily extendable framework for PINNs with clean separation of concerns
- **Gradient Normalization**: Improved training stability using advanced gradient normalization techniques
- **Multiple PDE Support**:
  - Heat Equation (1D)
  - Schrödinger Equation (1D)
  - Korteweg-de Vries (KdV) Equation
  - Navier-Stokes Equations for Crystal Growth (2D)
- **Neural Network Options**:
  - MLP (Multi-Layer Perceptron)
  - SIREN (Sinusoidal Representation Networks)
- **Visualization Tools**: Comprehensive plotting utilities for solution visualization

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib
- SciPy
- scikit-learn

## Installation

```bash
# Clone the repository
git clone https://github.com/bm777/gnpinn.git
cd gn-pinn

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The framework can be used through the command line interface:

```bash
python main.py --equation [heat|schrodinger|kdv|crystal] [options]
```

### Command Line Arguments

- `--equation`: Type of equation to solve (heat, schrodinger, kdv, crystal)
- `--use_gn`: Use gradient normalization for improved training
- `--network_type`: Neural network type (mlp or siren)
- `--hidden_layers`: Number of hidden layers in the neural network
- `--neurons`: Number of neurons per hidden layer
- `--learning_rate`: Learning rate for optimization
- `--epochs`: Number of training epochs
- `--device`: Device to use (cpu or cuda)
- `--output_dir`: Directory to save results

For Navier-Stokes crystal growth simulation:
- `--viscosity`: Kinematic viscosity coefficient
- `--thermal_diffusivity`: Thermal diffusivity
- `--density`: Fluid density
- `--plot_resolution`: Resolution for visualization plots

### Examples

#### Solving the Heat Equation
```bash
python main.py --equation heat --use_gn --network_type siren --hidden_layers 4 --neurons 50 --epochs 5000
```

#### Solving the Schrödinger Equation
```bash
python main.py --equation schrodinger --use_gn --network_type siren --hidden_layers 5 --neurons 100 --epochs 8000
```

#### Solving the KdV Equation
```bash
python main.py --equation kdv --use_gn --network_type siren --hidden_layers 5 --neurons 100 --epochs 10000
```

#### Crystal Growth Simulation using Navier-Stokes
```bash
python main.py --equation crystal --use_gn --network_type siren --hidden_layers 5 --neurons 100 --viscosity 0.01 --thermal_diffusivity 0.005 --epochs 5000
```

## Project Structure

```
gs-pinn/
├── core/                # Core components
├── models/              # Neural network models
│   ├── __init__.py
│   ├── mlp.py           # Multi-layer perceptron model
│   ├── siren.py         # SIREN model
│   └── pinn.py          # PINN and GNPINN implementations
├── equations/           # Equation implementations
│   ├── __init__.py
│   ├── base.py          # Base PDE class
│   ├── heat.py          # Heat equation
│   ├── schrodinger.py   # Schrödinger equation
│   ├── kdv.py           # KdV equation
│   └── navier_stokes.py # Navier-Stokes equations for crystal growth
├── utils/               # Utility functions
│   ├── __init__.py
│   └── helpers.py       # Helper functions
├── visualization/       # Visualization tools
│   ├── __init__.py
│   └── plotting.py      # Plotting utilities
├── examples/            # Example scripts
│   ├── __init__.py
│   ├── heat_equation.py     # Heat equation example
│   ├── schrodinger.py       # Schrödinger equation example
│   ├── kdv_equation.py      # KdV equation example
│   └── crystal_growth.py    # Crystal growth example using Navier-Stokes
├── main.py              # Main script
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## Extending the Framework

### Adding a New PDE

1. Create a new file in the `equations` directory
2. Implement a class that inherits from `PDEBase`
3. Implement the required methods:
   - `compute_residual`
   - `get_boundary_conditions`
   - `get_initial_conditions`
4. Update `equations/__init__.py` to include your new PDE

### Adding a New Neural Network Architecture

1. Create a new file in the `models` directory
2. Implement a class that inherits from `torch.nn.Module`
3. Update `models/__init__.py` to include your new model

## Future Work

- Add support for higher-dimensional problems
- Implement adaptive sampling strategies
- Add more visualization options
- Extend crystal growth modeling capabilities
- Add quantum state extensions for the Schrödinger equation
- Implement Allen-Cahn equation
- Implement 3D Navier-Stokes equations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the PINN framework developed by Raissi et al.
- Gradient normalization techniques inspired by Wang et al.
- SIREN implementation based on the paper by Sitzmann et al.

# Crystal Growth Simulation

This project simulates crystal growth using the Navier-Stokes equations and Physics-Informed Neural Networks (PINNs).

## Setup

The project uses a virtual environment named "gradient" for dependency management.

### Activating the Virtual Environment

```bash
# On macOS/Linux
source gradient/bin/activate

# On Windows
gradient\Scripts\activate
```

### Required Dependencies

After activating the virtual environment, ensure you have the following dependencies:

```bash
pip install numpy matplotlib torch
```

## Running the Simulation

### Simple Animation

To generate a series of frames showing the crystal growth:

```bash
python examples/simple_animation.py --frames 50
```

Options:
- `--frames`: Number of frames to generate (default: 50)
- `--output-dir`: Custom directory to save frames (optional)

After running the script, the frames will be saved in `results/simple_animation/frames/` and an HTML slideshow will be created at `results/simple_animation/slideshow.html`.

### Advanced Simulation with PINNs

For the full physics-informed neural network simulation:

```bash
python examples/crystal_growth_video.py
```

Options:
- `--frames`: Number of frames in the video
- `--duration`: Duration of the video in seconds
- `--fps`: Frames per second
- `--model`: Path to saved model (optional)
- `--train`: Force training a new model

## File Structure

- `equations/`: Contains the physical equations including Navier-Stokes
- `models/`: Neural network architectures (MLP, SIREN, GNPINN)
- `utils/`: Utility functions for training and visualization
- `examples/`: Example scripts
  - `simple_animation.py`: Creates frame-by-frame animation
  - `crystal_growth_video.py`: Full PINN-based simulation

## Visualization

The simulation creates both individual frames and an HTML slideshow for viewing the results. Open the slideshow in a web browser to see the animation with playback controls. 
