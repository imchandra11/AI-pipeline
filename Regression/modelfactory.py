"""
Neural network model factory for regression tasks.

This module provides a flexible feedforward neural network architecture
that can be configured via hyperparameters.
"""

from typing import Optional
import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    """
    Flexible feedforward neural network for regression.
    
    Supports configurable:
    - Input dimension (auto-detected from data)
    - Hidden layers (list of sizes)
    - Dropout rates (list matching hidden layers)
    - Activation functions
    - Output dimension (1 for regression)
    
    Args:
        input_dim: Number of input features
        hidden_layers: List of hidden layer sizes, e.g., [128, 64, 32]
        dropout_rates: List of dropout rates matching hidden layers, e.g., [0.15, 0.1, 0.05]
        activation: Activation function name ('relu', 'tanh', 'gelu', etc.)
        output_dim: Output dimension (default: 1 for regression)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        dropout_rates: Optional[list[float]] = None,
        activation: str = "relu",
        output_dim: int = 1,
    ):
        """
        Initialize the regression model.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rates: List of dropout rates (must match hidden_layers length)
            activation: Activation function name
            output_dim: Output dimension (default: 1)
        """
        super().__init__()
        
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        
        if not hidden_layers:
            raise ValueError("hidden_layers cannot be empty")
        
        if dropout_rates is None:
            dropout_rates = [0.0] * len(hidden_layers)
        
        if len(dropout_rates) != len(hidden_layers):
            raise ValueError(
                f"dropout_rates length ({len(dropout_rates)}) must match "
                f"hidden_layers length ({len(hidden_layers)})"
            )
        
        # Validate dropout rates
        for i, rate in enumerate(dropout_rates):
            if not 0 <= rate < 1:
                raise ValueError(f"dropout_rates[{i}] must be in [0, 1), got {rate}")
        
        # Get activation function
        activation_fn = self._get_activation(activation)
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_layers, dropout_rates)):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(activation_fn)
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def _get_activation(self, activation: str) -> nn.Module:
        """
        Get activation function module.
        
        Args:
            activation: Activation function name
            
        Returns:
            Activation function module
        """
        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
        }
        
        activation_lower = activation.lower()
        if activation_lower not in activation_map:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Supported: {list(activation_map.keys())}"
            )
        
        return activation_map[activation_lower]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.model(x)
    
    def get_input_dim(self) -> int:
        """Get the input dimension."""
        return self.input_dim
    
    def get_output_dim(self) -> int:
        """Get the output dimension."""
        return self.output_dim

