import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable architecture
    """
    
    def __init__(self, input_dim, output_dim, hidden_layers=4, neurons_per_layer=50, activation='tanh'):
        """
        Initialize the MLP.
        
        Args:
            input_dim: Dimension of input (number of features)
            output_dim: Dimension of output
            hidden_layers: Number of hidden layers
            neurons_per_layer: Number of neurons per hidden layer
            activation: Activation function to use (string or callable)
        """
        super(MLP, self).__init__()
        
        # Convert string activation to function
        if isinstance(activation, str):
            if activation.lower() == 'tanh':
                act_fn = nn.Tanh
            elif activation.lower() == 'relu':
                act_fn = nn.ReLU
            elif activation.lower() == 'gelu':
                act_fn = nn.GELU
            elif activation.lower() == 'sine':
                # Create a simple sine activation function
                class SineActivation(nn.Module):
                    def forward(self, x):
                        return torch.sin(x)
                act_fn = SineActivation
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
        else:
            act_fn = activation
        
        # Build network
        layers = [nn.Linear(input_dim, neurons_per_layer), act_fn()]
        
        for _ in range(hidden_layers):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(act_fn())
            
        layers.append(nn.Linear(neurons_per_layer, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.layers(x)


class SirenLayer(nn.Module):
    """
    SIREN layer with sine activation and special initialization
    """
    
    def __init__(self, in_features, out_features, is_first=False, omega=30.0):
        super(SirenLayer, self).__init__()
        self.omega = omega
        self.is_first = is_first
        
        self.linear = nn.Linear(in_features, out_features)
        
        # Initialize weights according to SIREN paper
        with torch.no_grad():
            if self.is_first:
                bound = 1 / in_features
            else:
                bound = torch.sqrt(torch.tensor(6.0 / in_features)) / self.omega
                
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)
            
    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class SIREN(nn.Module):
    """
    SIREN (Sinusoidal Representation Networks) model
    https://arxiv.org/abs/2006.09661
    
    Good for representing complex signals with implicit neural networks
    """
    
    def __init__(self, input_dim, output_dim, hidden_layers=4, neurons_per_layer=256, omega=30.0):
        super(SIREN, self).__init__()
        
        layers = []
        layers.append(SirenLayer(input_dim, neurons_per_layer, is_first=True, omega=omega))
        
        for _ in range(hidden_layers):
            layers.append(SirenLayer(neurons_per_layer, neurons_per_layer, omega=omega))
            
        self.layers = nn.Sequential(*layers)
        self.final_layer = nn.Linear(neurons_per_layer, output_dim)
        
    def forward(self, x):
        x = self.layers(x)
        x = self.final_layer(x)
        return x 