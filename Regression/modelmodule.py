"""
Generic PyTorch Lightning Module for regression tasks.

This module provides a reusable Lightning module that handles training,
validation, and testing steps for regression models.
"""

from typing import Optional, Any
import torch
import torch.nn as nn
import lightning as L
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from Regression.modelfactory import RegressionModel


class ModelModuleRGS(L.LightningModule):
    """
    Generic PyTorch Lightning Module for regression.
    
    Handles:
    - Training/validation/test steps
    - Loss calculation (MSE)
    - Optimizer and scheduler configuration
    - Metrics logging
    
    Args:
        model: Regression model instance
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        lr_scheduler_factor: Factor for ReduceLROnPlateau scheduler
        lr_scheduler_patience: Patience for ReduceLROnPlateau scheduler
        save_dir: Directory to save model artifacts
        name: Model name for saving
    """
    
    def __init__(
        self,
        model: RegressionModel,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler_factor: Optional[float] = None,
        lr_scheduler_patience: Optional[int] = None,
        save_dir: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize the Lightning module.
        
        Args:
            model: Regression model instance
            lr: Learning rate
            weight_decay: Weight decay
            lr_scheduler_factor: LR scheduler factor
            lr_scheduler_patience: LR scheduler patience
            save_dir: Save directory
            name: Model name
        """
        super().__init__()
        
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.save_dir = save_dir
        self.name = name
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # For logging
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        return self.model(x)
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch of (features, targets)
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat.squeeze(), y)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Calculate additional metrics
        mae = torch.mean(torch.abs(y_hat.squeeze() - y))
        self.log("train_mae", mae, on_step=False, on_epoch=True, logger=True)
        
        return loss
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        
        Args:
            batch: Batch of (features, targets)
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat.squeeze(), y)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Calculate additional metrics
        mae = torch.mean(torch.abs(y_hat.squeeze() - y))
        self.log("val_mae", mae, on_step=False, on_epoch=True, logger=True)
        
        return loss
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Test step.
        
        Args:
            batch: Batch of (features, targets)
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat.squeeze(), y)
        
        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        
        # Calculate additional metrics
        mae = torch.mean(torch.abs(y_hat.squeeze() - y))
        self.log("test_mae", mae, on_step=False, on_epoch=True, logger=True)
        
        return loss
    
    def configure_optimizers(self) -> dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # Optimizer will be set by Lightning CLI from config
        # This is a fallback if not provided via CLI
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Scheduler configuration
        if self.lr_scheduler_factor is not None and self.lr_scheduler_patience is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }
        
        return {"optimizer": optimizer}
    
    def get_model(self) -> RegressionModel:
        """Get the underlying model."""
        return self.model

