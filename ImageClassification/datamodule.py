"""
Generic PyTorch Lightning DataModule for image classification tasks.

This module provides a reusable data module that handles image loading,
preprocessing, augmentation, and train/val/test splits for image classification.
"""

from typing import Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
import joblib

from ImageClassification.dataset import ImageClassificationDataset


class DataModuleIMG(L.LightningDataModule):
    """
    Generic PyTorch Lightning DataModule for image classification.
    
    Handles:
    - Image loading from folder structure or CSV
    - Train/val/test splits
    - Image preprocessing and augmentation
    - Auto-detection of number of classes
    - Label encoding and persistence
    
    Args:
        data_dir: Directory containing class folders (folder-based) or None if using CSV
        csv_path: Path to CSV file with image_path and label columns (CSV-based)
        image_path_col: Column name for image paths in CSV (default: 'image_path')
        label_col: Column name for labels in CSV (default: 'label')
        image_size: Tuple of (height, width) for resizing images
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        val_split: Validation split ratio (0.0 to 1.0)
        test_split: Test split ratio (0.0 to 1.0), optional
        random_seed: Random seed for reproducibility
        augmentation: Dictionary with augmentation settings
        normalization: Dictionary with normalization mean and std
        save_preprocessor: Whether to save label encoder
        preprocessor_path: Path to save/load label encoder
    """
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        csv_path: Optional[Union[str, Path]] = None,
        image_path_col: str = "image_path",
        label_col: str = "label",
        image_size: tuple[int, int] = (224, 224),
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        test_split: Optional[float] = None,
        random_seed: int = 42,
        augmentation: Optional[dict] = None,
        normalization: Optional[dict] = None,
        save_preprocessor: bool = True,
        preprocessor_path: Optional[str] = None,
    ):
        """
        Initialize the image data module.
        
        Args:
            data_dir: Directory with class folders OR None if using CSV
            csv_path: Path to CSV file OR None if using folder structure
            image_path_col: Column name for image paths in CSV
            label_col: Column name for labels in CSV
            image_size: (height, width) for image resizing
            batch_size: Batch size
            num_workers: Number of workers
            val_split: Validation split ratio
            test_split: Test split ratio (optional)
            random_seed: Random seed
            augmentation: Augmentation config dict
            normalization: Normalization config dict
            save_preprocessor: Whether to save label encoder
            preprocessor_path: Path to save label encoder
        """
        super().__init__()
        
        if data_dir is None and csv_path is None:
            raise ValueError("Either data_dir or csv_path must be provided")
        if data_dir is not None and csv_path is not None:
            raise ValueError("Cannot specify both data_dir and csv_path. Use one or the other.")
        
        self.data_dir = Path(data_dir) if data_dir else None
        self.csv_path = Path(csv_path) if csv_path else None
        self.image_path_col = image_path_col
        self.label_col = label_col
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.augmentation = augmentation or {}
        self.normalization = normalization or {}
        self.save_preprocessor = save_preprocessor
        self.preprocessor_path = preprocessor_path
        
        # Validate splits
        if not 0 < val_split < 1:
            raise ValueError(f"val_split must be in (0, 1), got {val_split}")
        
        if test_split is not None and not 0 < test_split < 1:
            raise ValueError(f"test_split must be in (0, 1), got {test_split}")
        
        if val_split + (test_split or 0) >= 1:
            raise ValueError(
                f"val_split ({val_split}) + test_split ({test_split or 0}) must be < 1"
            )
        
        # Will be set during setup
        self.label_encoder: Optional[LabelEncoder] = None
        self.num_classes: Optional[int] = None
        self.class_names: Optional[list[str]] = None
        self.class_to_idx: Optional[dict[str, int]] = None
        self.train_dataset: Optional[ImageClassificationDataset] = None
        self.val_dataset: Optional[ImageClassificationDataset] = None
        self.test_dataset: Optional[ImageClassificationDataset] = None
        
        # Transforms
        self.train_transform = None
        self.val_transform = None
        self.test_transform = None
    
    def prepare_data(self):
        """Download or prepare data (called only on main process)."""
        # Validate data directory or CSV exists
        if self.data_dir is not None:
            if not self.data_dir.exists():
                raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        elif self.csv_path is not None:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training, validation, and testing.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Load image paths and labels
        if self.data_dir is not None:
            image_paths, labels, class_to_idx = self._load_from_folders()
        else:
            image_paths, labels, class_to_idx = self._load_from_csv()
        
        # Encode labels to 0-indexed integers
        unique_classes = sorted(set(labels))
        
        # Create label encoder if labels are strings
        if labels and isinstance(labels[0], str):
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
            self.class_names = self.label_encoder.classes_.tolist()
        else:
            # Labels are already integers
            self.label_encoder = None
            encoded_labels = np.array(labels)
            
            # Store original unique labels before mapping
            original_unique_labels = sorted(np.unique(labels))
            
            # Ensure 0-indexed
            min_label = min(encoded_labels)
            if min_label != 0:
                encoded_labels = encoded_labels - min_label
            
            # Get unique class names from original labels (before 0-indexing)
            self.class_names = [str(c) for c in original_unique_labels]
        
        # Update class_to_idx with encoded indices
        if self.label_encoder:
            # Map original class names to encoded indices
            self.class_to_idx = {
                class_name: self.label_encoder.transform([class_name])[0]
                for class_name in class_to_idx.keys()
            }
        else:
            # Map original labels to 0-indexed labels
            original_unique_labels = sorted(set(labels))
            label_mapping = {orig: idx for idx, orig in enumerate(original_unique_labels)}
            self.class_to_idx = {
                str(k): label_mapping[k] if k in label_mapping else label_mapping[int(k)]
                for k in class_to_idx.keys()
            }
        
        self.num_classes = len(unique_classes)
        
        # Create transforms
        self._create_transforms()
        
        # Split data
        if self.test_split:
            # Train+Val, Test split
            train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
                image_paths,
                encoded_labels,
                test_size=self.test_split,
                random_state=self.random_seed,
                shuffle=True,
                stratify=encoded_labels if self.num_classes > 1 else None,
            )
            # Train, Val split
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_val_paths,
                train_val_labels,
                test_size=self.val_split / (1 - self.test_split),
                random_state=self.random_seed,
                shuffle=True,
                stratify=train_val_labels if self.num_classes > 1 else None,
            )
        else:
            # Train, Val split only
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                image_paths,
                encoded_labels,
                test_size=self.val_split,
                random_state=self.random_seed,
                shuffle=True,
                stratify=encoded_labels if self.num_classes > 1 else None,
            )
            test_paths, test_labels = val_paths, val_labels
        
        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = ImageClassificationDataset(
                train_paths, train_labels, self.train_transform, self.class_to_idx
            )
            self.val_dataset = ImageClassificationDataset(
                val_paths, val_labels, self.val_transform, self.class_to_idx
            )
            
            # Save label encoder if requested
            if self.save_preprocessor and self.preprocessor_path and self.label_encoder:
                preprocessor_dir = Path(self.preprocessor_path).parent
                preprocessor_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.label_encoder, self.preprocessor_path)
        
        if stage == "test" or stage is None:
            if self.test_split:
                self.test_dataset = ImageClassificationDataset(
                    test_paths, test_labels, self.test_transform, self.class_to_idx
                )
            else:
                # Use validation set as test if no test split
                self.test_dataset = self.val_dataset
    
    def _load_from_folders(self) -> tuple[list[Path], list[str], dict[str, int]]:
        """Load images from folder structure (class folders)."""
        class_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        if not class_folders:
            raise ValueError(f"No class folders found in {self.data_dir}")
        
        image_paths = []
        labels = []
        class_to_idx = {folder.name: idx for idx, folder in enumerate(class_folders)}
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        for class_folder in class_folders:
            class_name = class_folder.name
            class_images = [
                img for img in class_folder.iterdir()
                if img.suffix.lower() in image_extensions
            ]
            
            if not class_images:
                raise ValueError(f"No images found in class folder: {class_folder}")
            
            image_paths.extend(class_images)
            labels.extend([class_name] * len(class_images))
        
        return image_paths, labels, class_to_idx
    
    def _load_from_csv(self) -> tuple[list[Path], list, dict[str, int]]:
        """Load images from CSV file."""
        df = pd.read_csv(self.csv_path)
        
        if self.image_path_col not in df.columns:
            raise ValueError(f"Column '{self.image_path_col}' not found in CSV")
        if self.label_col not in df.columns:
            raise ValueError(f"Column '{self.label_col}' not found in CSV")
        
        # Convert image paths to absolute paths relative to CSV location
        csv_dir = self.csv_path.parent
        image_paths = [
            csv_dir / Path(path) if not Path(path).is_absolute() else Path(path)
            for path in df[self.image_path_col]
        ]
        
        labels = df[self.label_col].tolist()
        
        # Create class_to_idx from unique labels
        unique_labels = sorted(set(labels))
        class_to_idx = {str(label): idx for idx, label in enumerate(unique_labels)}
        
        return image_paths, labels, class_to_idx
    
    def _create_transforms(self):
        """Create train, validation, and test transforms."""
        # Normalization
        mean = self.normalization.get('mean', [0.485, 0.456, 0.406])  # ImageNet defaults
        std = self.normalization.get('std', [0.229, 0.224, 0.225])
        
        # Base transform (always applied)
        base_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        # Validation/Test transform (no augmentation)
        self.val_transform = base_transform
        self.test_transform = base_transform
        
        # Training transform (with augmentation if enabled)
        if self.augmentation.get('enabled', True):
            aug_transforms = []
            
            # Random rotation
            if 'rotation' in self.augmentation:
                aug_transforms.append(
                    transforms.RandomRotation(degrees=self.augmentation['rotation'])
                )
            
            # Random horizontal flip
            if self.augmentation.get('horizontal_flip', True):
                aug_transforms.append(transforms.RandomHorizontalFlip())
            
            # Random vertical flip
            if self.augmentation.get('vertical_flip', False):
                aug_transforms.append(transforms.RandomVerticalFlip())
            
            # Color jitter
            if 'color_jitter' in self.augmentation:
                brightness = self.augmentation.get('color_jitter_brightness', 0.2)
                contrast = self.augmentation.get('color_jitter_contrast', 0.2)
                saturation = self.augmentation.get('color_jitter_saturation', 0.2)
                hue = self.augmentation.get('color_jitter_hue', 0.1)
                aug_transforms.append(
                    transforms.ColorJitter(
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        hue=hue
                    )
                )
            
            # Random crop
            if 'random_crop' in self.augmentation:
                crop_size = self.augmentation['random_crop']
                aug_transforms.append(transforms.RandomCrop(crop_size))
            
            # Resize (always first)
            aug_transforms.insert(0, transforms.Resize(self.image_size))
            
            # ToTensor and Normalize
            aug_transforms.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            
            self.train_transform = transforms.Compose(aug_transforms)
        else:
            # No augmentation
            self.train_transform = base_transform
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset not set. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        if self.val_dataset is None:
            raise RuntimeError("val_dataset not set. Call setup('fit') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if self.test_dataset is None:
            raise RuntimeError("test_dataset not set. Call setup('test') first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def get_num_classes(self) -> int:
        """
        Get the number of classes.
        
        Returns:
            Number of classes
        """
        if self.num_classes is None:
            raise RuntimeError("num_classes not set. Call setup() first.")
        return self.num_classes
    
    def get_class_names(self) -> list[str]:
        """
        Get the class names.
        
        Returns:
            List of class names in order (0-indexed)
        """
        if self.class_names is None:
            raise RuntimeError("class_names not set. Call setup() first.")
        return self.class_names
    
    def get_class_to_idx(self) -> dict[str, int]:
        """
        Get the class name to index mapping.
        
        Returns:
            Dictionary mapping class names to indices
        """
        if self.class_to_idx is None:
            raise RuntimeError("class_to_idx not set. Call setup() first.")
        return self.class_to_idx.copy()
    
    def get_label_encoder(self):
        """
        Get the label encoder if labels were categorical.
        
        Returns:
            LabelEncoder instance or None if labels were already integers
        """
        return self.label_encoder

