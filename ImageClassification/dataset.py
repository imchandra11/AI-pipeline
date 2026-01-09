"""
Generic PyTorch Dataset for image classification data.

This module provides a reusable dataset class that handles image loading,
transformation, and target extraction for image classification tasks.
"""

from typing import Optional, Union, Callable
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np


class ImageClassificationDataset(Dataset):
    """
    PyTorch Dataset for image classification data.
    
    Supports two data formats:
    1. Folder-based: images organized in class folders (data/class1/, data/class2/)
    2. CSV-based: CSV file with image_path and label columns
    
    Handles image loading, transformation, and label encoding.
    
    Args:
        image_paths: List of image file paths
        labels: List of integer labels (0-indexed)
        transform: Optional torchvision transform to apply
        class_to_idx: Optional mapping from class names to indices
    """
    
    def __init__(
        self,
        image_paths: list[Union[str, Path]],
        labels: list[int],
        transform: Optional[Callable] = None,
        class_to_idx: Optional[dict[str, int]] = None,
    ):
        """
        Initialize the image classification dataset.
        
        Args:
            image_paths: List of paths to image files
            labels: List of integer class labels (0-indexed)
            transform: Optional transform function to apply to images
            class_to_idx: Optional mapping from class names to label indices
        """
        if len(image_paths) != len(labels):
            raise ValueError(
                f"Number of image paths ({len(image_paths)}) must match "
                f"number of labels ({len(labels)})"
            )
        
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels
        self.transform = transform
        self.class_to_idx = class_to_idx or {}
        
        # Convert labels to tensor
        self.labels_tensor = torch.LongTensor(self.labels)
        
        # Validate image paths exist
        for img_path in self.image_paths:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: Union[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label) as tensors
        """
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
        
        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)
        
        # Get label
        label = self.labels_tensor[idx]
        
        return image, label
    
    def get_class_to_idx(self) -> dict[str, int]:
        """Get the class name to index mapping."""
        return self.class_to_idx.copy()
    
    def get_idx_to_class(self) -> dict[int, str]:
        """Get the index to class name mapping."""
        return {idx: name for name, idx in self.class_to_idx.items()}

