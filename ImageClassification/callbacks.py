"""
ONNX export callback for PyTorch Lightning image classification models.

This module provides a callback to export trained image classification models
to ONNX format for production deployment.
"""

import logging
from pathlib import Path
from typing import Optional
import torch
import lightning as L

# Set up logger
logger = logging.getLogger(__name__)


class ONNXExportCallback(L.pytorch.callbacks.Callback):
    """
    Callback to export image classification model to ONNX format after training.
    
    The callback exports the model after training completes, using
    the input shape from the datamodule or provided configuration.
    
    Args:
        output_dir: Directory to save ONNX model
        model_name: Name for the saved model file
        input_shape: Input shape as [channels, height, width] (if None, will try to get from datamodule)
    """
    
    def __init__(
        self,
        output_dir: str = "models",
        model_name: str = "model",
        input_shape: Optional[list[int]] = None,
    ):
        """
        Initialize the ONNX export callback.
        
        Args:
            output_dir: Output directory
            model_name: Model name
            input_shape: Input shape [C, H, W] (auto-detected if None)
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.input_shape = input_shape
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Export model to ONNX format after training ends.
        
        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        # Get input shape
        input_shape = self.input_shape
        if input_shape is None:
            # Try to get from model first
            if hasattr(pl_module.model, 'get_input_channels'):
                input_channels = pl_module.model.get_input_channels()
                # Try to get image size from datamodule
                if hasattr(trainer.datamodule, 'image_size'):
                    image_size = trainer.datamodule.image_size
                    input_shape = [input_channels, image_size[0], image_size[1]]
                else:
                    # Default to 224x224 if not available
                    input_shape = [input_channels, 224, 224]
            else:
                # Fallback: try datamodule
                if hasattr(trainer.datamodule, 'image_size'):
                    image_size = trainer.datamodule.image_size
                    input_channels = 3  # Default to RGB
                    input_shape = [input_channels, image_size[0], image_size[1]]
                else:
                    raise ValueError(
                        "Could not determine input_shape. Please provide it in callback init_args "
                        "as [channels, height, width]."
                    )
        
        if len(input_shape) != 3:
            raise ValueError(
                f"input_shape must be [channels, height, width], got {input_shape}"
            )
        
        channels, height, width = input_shape
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the underlying PyTorch model
        model = pl_module.model
        model.eval()
        
        # Create dummy input (batch_size=1)
        dummy_input = torch.randn(1, channels, height, width)
        
        # Export to ONNX
        onnx_path = self.output_dir / f"{self.model_name}.onnx"
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
            )
            
            # Log success using Python's logging module
            success_msg = f"Model exported to ONNX: {onnx_path}"
            logger.info(success_msg)
            print(f"✓ {success_msg}")
        
        except Exception as e:
            # Log error using Python's logging module
            error_msg = f"Failed to export model to ONNX: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"✗ {error_msg}")
            raise

