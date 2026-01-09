"""
CNN model factory for image classification tasks.

This module provides flexible CNN architectures that can be configured
via hyperparameters for image classification.
"""

from typing import Optional
import torch
import torch.nn as nn


class ImageClassificationModel(nn.Module):
    """
    Flexible CNN for image classification.
    
    Supports configurable:
    - Input channels (1 for grayscale, 3 for RGB)
    - CNN architecture (simple, medium, deep, or custom)
    - Number of output classes
    - Dropout rates
    - Activation functions
    
    Args:
        input_channels: Number of input channels (1 or 3)
        num_classes: Number of output classes
        architecture: Architecture preset ('simple', 'medium', 'deep') or 'custom'
        conv_layers: Custom conv layer configs (if architecture='custom')
        hidden_dims: FC layer sizes after conv layers
        dropout_rates: Dropout rates for FC layers
        activation: Activation function name
        input_size: Input image size (height, width) for calculating FC input
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 0,
        architecture: str = "medium",
        conv_layers: Optional[list[dict]] = None,
        hidden_dims: Optional[list[int]] = None,
        dropout_rates: Optional[list[float]] = None,
        activation: str = "relu",
        input_size: tuple[int, int] = (224, 224),
    ):
        """
        Initialize the CNN model.
        
        Args:
            input_channels: Number of input channels (1 or 3)
            num_classes: Number of output classes (can be 0 if will be set later)
            architecture: Architecture preset or 'custom'
            conv_layers: Custom conv layer configurations
            hidden_dims: FC layer sizes
            dropout_rates: Dropout rates for FC layers
            activation: Activation function name
            input_size: Input image size (H, W)
        """
        super().__init__()
        
        if input_channels not in [1, 3]:
            raise ValueError(f"input_channels must be 1 or 3, got {input_channels}")
        
        if num_classes < 0:
            raise ValueError(f"num_classes must be non-negative, got {num_classes}")
        
        # Store configuration
        self._input_channels = input_channels
        self._num_classes = num_classes
        self._architecture = architecture
        self._input_size = input_size
        self._activation = activation
        
        # Build conv layers
        if architecture == "custom":
            if conv_layers is None:
                raise ValueError("conv_layers must be provided when architecture='custom'")
            self.conv_layers = self._build_custom_conv_layers(conv_layers, input_channels)
        else:
            self.conv_layers = self._build_preset_conv_layers(architecture, input_channels)
        
        # Calculate feature map size after conv layers
        self._feature_size = self._calculate_feature_size()
        
        # Build FC layers
        hidden_dims = hidden_dims or []
        dropout_rates = dropout_rates or [0.0] * len(hidden_dims)
        
        if len(dropout_rates) != len(hidden_dims):
            raise ValueError(
                f"dropout_rates length ({len(dropout_rates)}) must match "
                f"hidden_dims length ({len(hidden_dims)})"
            )
        
        # Only build model if num_classes is set
        if num_classes > 0:
            self._build_model(hidden_dims, dropout_rates)
        else:
            # Create placeholder - will be built when num_classes is set
            self.model = None
            self.input_channels = input_channels
            self.num_classes = num_classes
    
    def _build_preset_conv_layers(self, architecture: str, input_channels: int) -> nn.Sequential:
        """Build conv layers from preset architecture."""
        layers = []
        in_channels = input_channels
        
        if architecture == "simple":
            # Simple: 2 conv blocks
            configs = [
                {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            ]
        elif architecture == "medium":
            # Medium: 4 conv blocks
            configs = [
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            ]
        elif architecture == "deep":
            # Deep: 6 conv blocks
            configs = [
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            ]
        else:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                f"Supported: 'simple', 'medium', 'deep', 'custom'"
            )
        
        for config in configs:
            out_channels = config['out_channels']
            kernel_size = config['kernel_size']
            stride = config.get('stride', 1)
            padding = config.get('padding', 0)
            
            # Conv + BN + ReLU + MaxPool
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _build_custom_conv_layers(self, conv_layers: list[dict], input_channels: int) -> nn.Sequential:
        """Build conv layers from custom configuration."""
        layers = []
        in_channels = input_channels
        
        for layer_config in conv_layers:
            out_channels = layer_config['out_channels']
            kernel_size = layer_config.get('kernel_size', 3)
            stride = layer_config.get('stride', 1)
            padding = layer_config.get('padding', 1)
            pool = layer_config.get('pool', True)
            
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
            if pool:
                pool_size = layer_config.get('pool_size', 2)
                pool_stride = layer_config.get('pool_stride', 2)
                layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride))
            
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _calculate_feature_size(self) -> int:
        """Calculate feature map size after conv layers."""
        # Create dummy input
        dummy_input = torch.randn(1, self._input_channels, self._input_size[0], self._input_size[1])
        
        # Forward through conv layers only
        with torch.no_grad():
            x = dummy_input
            x = self.conv_layers(x)
            feature_size = x.numel() // x.size(0)  # Total features per sample
        
        return feature_size
    
    def _build_model(self, hidden_dims: list[int], dropout_rates: list[float]):
        """Build the complete model (conv + FC layers)."""
        activation_fn = self._get_activation(self._activation)
        
        # Build FC layers
        fc_layers = []
        prev_dim = self._feature_size
        
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            fc_layers.append(nn.BatchNorm1d(hidden_dim))
            fc_layers.append(activation_fn)
            
            if dropout_rate > 0:
                fc_layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation - outputs logits for CrossEntropyLoss)
        fc_layers.append(nn.Linear(prev_dim, self._num_classes))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Combine conv and FC
        self.model = nn.Sequential(self.conv_layers, nn.Flatten(), self.fc_layers)
        
        self.input_channels = self._input_channels
        self.num_classes = self._num_classes
    
    def set_num_classes(self, num_classes: int):
        """
        Set number of classes and build the model.
        
        Args:
            num_classes: Number of output classes
        """
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        
        if self.model is not None:
            raise RuntimeError("Model already built. Cannot change num_classes.")
        
        self._num_classes = num_classes
        
        # Get default hidden_dims and dropout_rates based on architecture
        if self._architecture == "simple":
            hidden_dims = [128, 64]
            dropout_rates = [0.5, 0.3]
        elif self._architecture == "medium":
            hidden_dims = [512, 256]
            dropout_rates = [0.5, 0.3]
        elif self._architecture == "deep":
            hidden_dims = [512, 256, 128]
            dropout_rates = [0.5, 0.3, 0.2]
        else:
            # Custom - use defaults
            hidden_dims = [512, 256]
            dropout_rates = [0.5, 0.3]
        
        self._build_model(hidden_dims, dropout_rates)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module."""
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
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes) - logits (no softmax)
        """
        if self.model is None:
            raise RuntimeError(
                "Model not built. num_classes must be set before forward pass."
            )
        return self.model(x)
    
    def get_input_channels(self) -> int:
        """Get the number of input channels."""
        return self.input_channels
    
    def get_num_classes(self) -> int:
        """Get the number of classes."""
        return self.num_classes

