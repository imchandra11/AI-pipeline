# AI Model Training Pipeline - Image Classification Module

A generic, modular, and reusable PyTorch Lightning pipeline for training image classification models using CNNs. This pipeline is fully config-driven, allowing you to train models on any image classification dataset by simply modifying a YAML configuration file.

## Features

- **Fully Config-Driven**: All settings (architecture, hyperparameters, paths) controlled via YAML files
- **Generic & Reusable**: Use the same codebase for any image classification task (animals, products, documents, etc.)
- **Flexible Data Formats**: Supports both folder-based organization (class folders) and CSV-based datasets
- **Auto-Dimension Detection**: Automatically detects number of classes from dataset structure
- **Configurable CNN Architectures**: Choose from preset architectures (simple, medium, deep) or define custom layers
- **Data Augmentation**: Built-in augmentation support for better generalization
- **Production-Ready**: Exports models to ONNX format for easy deployment
- **PyTorch Lightning**: Built on PyTorch Lightning for scalable, professional ML training
- **Comprehensive Metrics**: Tracks Accuracy, F1-Score, Precision, and Recall (macro-averaged)

## Project Structure

```
AI-pipeline/
├── ImageClassification/              # Generic image classification module (reusable)
│   ├── __init__.py
│   ├── cli.py                        # Custom Lightning CLI
│   ├── main.py                       # Standard CLI entry point
│   ├── mainfittest.py                # Fit+test workflow entry point
│   ├── dataset.py                    # PyTorch Dataset for images
│   ├── datamodule.py                 # Lightning DataModule
│   ├── modelfactory.py               # CNN model factory
│   ├── modelmodule.py                # Lightning Module for training
│   └── callbacks.py                  # ONNX export callback
├── YourProjectName/                   # Example project (config-only)
│   ├── configs/
│   │   ├── your_project.yaml         # Main configuration
│   │   └── your_project.local.yaml  # Local overrides
│   ├── data/
│   │   └── images/                   # Image dataset
│   │       ├── class1/
│   │       │   ├── img1.jpg
│   │       │   └── img2.jpg
│   │       ├── class2/
│   │       │   ├── img3.jpg
│   │       │   └── img4.jpg
│   │       └── ...
│   ├── models/                       # Output: trained models, label encoders
│   └── lightning_logs/               # Output: training logs
├── Classification/                    # Tabular classification module
├── Regression/                        # Regression module
├── requirements.txt                  # Python dependencies
└── README_IMAGE_CLASSIFICATION.md    # This file
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: The requirements include `torchvision` and `Pillow` for image processing.

### 3. Verify Installation

```bash
python -c "import lightning as L; import torch; import torchvision; from PIL import Image; print('Installation successful!')"
```

## Creating a New Image Classification Project

To train a model on a new image dataset, follow these steps:

### Step 1: Create Project Directory Structure

```bash
mkdir YourProjectName
mkdir YourProjectName/configs
mkdir YourProjectName/data
mkdir YourProjectName/data/images
mkdir YourProjectName/models
mkdir YourProjectName/lightning_logs
```

### Step 2: Organize Your Image Dataset

You have two options for organizing your images:

#### Option A: Folder-Based Organization (Recommended)

Organize images in class folders:

```
YourProjectName/data/images/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── class3/
    ├── image1.jpg
    └── ...
```

**Supported formats**: JPG, JPEG, PNG, BMP, TIFF, WEBP

#### Option B: CSV-Based Organization

Create a CSV file with image paths and labels:

```csv
image_path,label
images/cat/cat1.jpg,cat
images/cat/cat2.jpg,cat
images/dog/dog1.jpg,dog
images/dog/dog2.jpg,dog
```

### Step 3: Create Configuration File

Create `YourProjectName/configs/your_project.yaml`:

```yaml
# Image Classification Model Configuration
# lightning.pytorch==2.1.0
seed_everything: true

trainer:
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        filename: "{epoch}-{val_acc:.2f}.best"
        monitor: "val_acc"
        mode: "max"
        save_top_k: 1
        verbose: true
        save_on_train_epoch_end: false
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        filename: "{epoch}.last"
        monitor: "step"
        mode: "max"
        save_top_k: 1
        verbose: true
        save_on_train_epoch_end: false
    - class_path: ImageClassification.callbacks.ONNXExportCallback
      init_args:
        output_dir: "YourProjectName/models"
        model_name: "your_model_name"
        input_shape: [3, 224, 224]  # [channels, height, width]

  logger:
    class_path: lightning.pytorch.loggers.TensorBoardLogger
    init_args:
      save_dir: "YourProjectName/lightning_logs"
      name: "YourProjectTraining"
      default_hp_metric: false

  max_epochs: 50
  num_sanity_val_steps: 2
  check_val_every_n_epoch: 1
  log_every_n_steps: 10
  accelerator: auto
  devices: auto
  precision: 16-mixed
  default_root_dir: "YourProjectName/lightning_logs/YourProjectTraining"

model:
  class_path: ImageClassification.modelmodule.ModelModuleIMG
  init_args:
    lr: 0.001
    weight_decay: 0.0001
    lr_scheduler_factor: 0.5
    lr_scheduler_patience: 5
    save_dir: "YourProjectName/models"
    name: "your_model_name"
    model:
      class_path: ImageClassification.modelfactory.ImageClassificationModel
      init_args:
        input_channels: 3  # 3 for RGB, 1 for grayscale
        num_classes: 0  # Auto-set from datamodule via CLI linking
        architecture: "medium"  # "simple", "medium", "deep", or "custom"
        # Optional: Custom architecture
        # conv_layers:
        #   - out_channels: 64
        #     kernel_size: 3
        #     pool: true
        #   - out_channels: 128
        #     kernel_size: 3
        #     pool: true
        hidden_dims: [512, 256]  # FC layer sizes (optional, defaults based on architecture)
        dropout_rates: [0.5, 0.3]  # Dropout for FC layers (optional)
        activation: "relu"  # "relu", "tanh", "gelu", "sigmoid", "leaky_relu", "elu"
        input_size: [224, 224]  # [height, width]

optimizer: 
  class_path: torch.optim.Adam
  init_args:
    lr: 0.001
    weight_decay: 0.0001

lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: 0.001
    pct_start: 0.1
    total_steps: 1000

data:
  class_path: ImageClassification.datamodule.DataModuleIMG
  init_args:
    # Option 1: Folder-based (recommended)
    data_dir: "YourProjectName/data/images"
    
    # Option 2: CSV-based (uncomment to use)
    # csv_path: "YourProjectName/data/image_labels.csv"
    # image_path_col: "image_path"
    # label_col: "label"
    
    image_size: [224, 224]  # [height, width]
    batch_size: 32
    num_workers: 4
    val_split: 0.2
    random_seed: 42
    augmentation:
      enabled: true
      rotation: 15
      horizontal_flip: true
      vertical_flip: false
      color_jitter: 0.2
      color_jitter_brightness: 0.2
      color_jitter_contrast: 0.2
      color_jitter_saturation: 0.2
      color_jitter_hue: 0.1
      # random_crop: 200  # Optional: crop size
    normalization:
      mean: [0.485, 0.456, 0.406]  # ImageNet defaults
      std: [0.229, 0.224, 0.225]
    save_preprocessor: true
    preprocessor_path: "YourProjectName/models/label_encoder.joblib"

fit:
  ckpt_path: null   # for resume training

test:
  ckpt_path: best   # checkpoint to use for test and predict
```

### Step 4: Create Local Override File (Optional)

Create `YourProjectName/configs/your_project.local.yaml` for local-specific settings:

```yaml
# Local configuration overrides
trainer:
  max_epochs: 10  # Quick test
  precision: 32   # Use full precision if mixed precision causes issues
  devices: 1      # Use single device

data:
  init_args:
    batch_size: 16  # Smaller batch for testing
    num_workers: 0  # No parallel loading for debugging
```

## Running the Project

**Important:** Always run commands from the project root directory (where `ImageClassification/` folder is located).

### Option 1: Fit + Test Workflow (Recommended for Quick Testing)

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "."; python ImageClassification/mainfittest.py --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

**Windows (CMD):**
```cmd
set PYTHONPATH=. && python ImageClassification/mainfittest.py --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

**Linux/Mac:**
```bash
PYTHONPATH=. python ImageClassification/mainfittest.py \
  --config YourProjectName/configs/your_project.yaml \
  --config YourProjectName/configs/your_project.local.yaml
```

### Option 2: Standard Lightning CLI Commands

**Training only (Windows PowerShell):**
```powershell
$env:PYTHONPATH = "."; python ImageClassification/main.py fit --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

**Testing only:**
```powershell
$env:PYTHONPATH = "."; python ImageClassification/main.py test --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

### Option 3: Using Jupyter Notebook

Create `YourProjectNameTrainer.ipynb`:

```python
# Run fit and test workflow
%run ImageClassification/mainfittest.py --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

## Configuration Reference

### Data Configuration

**Key Parameters:**
- `data_dir`: Path to folder containing class subdirectories (folder-based)
- `csv_path`: Path to CSV file with image paths and labels (CSV-based)
- `image_size`: Tuple of `[height, width]` for resizing images (e.g., `[224, 224]`)
- `batch_size`: Batch size for training (default: 32)
- `num_workers`: Number of parallel data loading workers (default: 4)
- `val_split`: Validation split ratio (0.0 to 1.0, default: 0.2)
- `augmentation`: Dictionary with augmentation settings (see below)
- `normalization`: Dictionary with normalization mean and std

**Augmentation Options:**
- `enabled`: Enable/disable augmentation (default: true)
- `rotation`: Random rotation degrees (default: 15)
- `horizontal_flip`: Random horizontal flip (default: true)
- `vertical_flip`: Random vertical flip (default: false)
- `color_jitter`: Color jitter intensity (default: 0.2)
- `color_jitter_brightness/contrast/saturation/hue`: Individual color jitter parameters
- `random_crop`: Crop size for random cropping (optional)

**Normalization:**
- `mean`: Mean values for normalization (default: ImageNet values)
- `std`: Standard deviation values (default: ImageNet values)

### Model Configuration

**Key Parameters:**
- `input_channels`: Number of input channels (3 for RGB, 1 for grayscale)
- `num_classes`: Number of output classes (auto-set from datamodule if 0)
- `architecture`: Architecture preset (`"simple"`, `"medium"`, `"deep"`, or `"custom"`)
- `conv_layers`: Custom conv layer configurations (if architecture='custom')
- `hidden_dims`: FC layer sizes after conv layers (optional, has defaults)
- `dropout_rates`: Dropout rates for FC layers (optional, has defaults)
- `activation`: Activation function (`"relu"`, `"tanh"`, `"gelu"`, `"sigmoid"`, `"leaky_relu"`, `"elu"`)
- `input_size`: Input image size `[height, width]` (default: `[224, 224]`)

**Architecture Presets:**

1. **Simple**: 2 conv blocks (32, 64 channels)
   - Good for: Small datasets, fast training, simple tasks
   
2. **Medium**: 4 conv blocks (64, 128, 256, 512 channels)
   - Good for: Medium datasets, balanced performance
   
3. **Deep**: 6 conv blocks (64, 128, 256, 256, 512, 512 channels)
   - Good for: Large datasets, complex tasks

**Custom Architecture Example:**
```yaml
architecture: "custom"
conv_layers:
  - out_channels: 32
    kernel_size: 3
    stride: 1
    padding: 1
    pool: true
    pool_size: 2
  - out_channels: 64
    kernel_size: 3
    stride: 1
    padding: 1
    pool: true
    pool_size: 2
```

### Trainer Configuration

**Key Parameters:**
- `max_epochs`: Number of training epochs
- `precision`: Training precision (`"16-mixed"`, `"32"`, `"bf16-mixed"`)
- `accelerator`: Hardware accelerator (`"auto"`, `"gpu"`, `"cpu"`)
- `devices`: Number of devices (`"auto"`, `1`, `[0, 1]`)

## Output Files

After training, you'll find:

1. **Models Directory** (`YourProjectName/models/`):
   - `your_model_name.onnx`: ONNX model for inference
   - `label_encoder.joblib`: Label encoder for mapping predictions back to class names

2. **Checkpoints** (`YourProjectName/lightning_logs/YourProjectTraining/version_X/checkpoints/`):
   - `epoch-X-val_acc=Y.best.ckpt`: Best model checkpoint
   - `epoch-X.last.ckpt`: Last epoch checkpoint

3. **Training Logs** (`YourProjectName/lightning_logs/`):
   - TensorBoard logs for visualization

## Viewing Training Progress

### TensorBoard

```bash
tensorboard --logdir YourProjectName/lightning_logs
```

Then open `http://localhost:6006` in your browser.

## Inference Example

After training, you can use the model for inference:

```python
import joblib
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

# Setup paths
MODELS_DIR = Path("YourProjectName/models")
IMAGE_SIZE = (224, 224)

# Load label encoder
label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

# Load ONNX model
session = ort.InferenceSession(str(MODELS_DIR / "your_model_name.onnx"))

# Prepare transforms (same as validation)
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess image
image_path = "path/to/your/image.jpg"
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)

# Predict
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: image_tensor})
logits = output[0][0]

# Get predicted class
predicted_class_idx = int(np.argmax(logits))
predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
confidence = float(np.exp(logits[predicted_class_idx]) / np.sum(np.exp(logits)))

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence * 100:.2f}%")
```

## Example Projects

### Animal Classification

Organize images:
```
AnimalClassification/data/images/
├── cat/
│   ├── cat1.jpg
│   └── cat2.jpg
├── dog/
│   ├── dog1.jpg
│   └── dog2.jpg
└── bird/
    ├── bird1.jpg
    └── bird2.jpg
```

Config:
```yaml
data:
  init_args:
    data_dir: "AnimalClassification/data/images"
    image_size: [224, 224]
    batch_size: 32

model:
  init_args:
    model:
      init_args:
        architecture: "medium"
```

### Product Classification

Similar structure with product categories.

## Troubleshooting

### Common Issues

**1. FileNotFoundError: Data directory not found**
- Check that `data_dir` path in config is correct relative to project root
- Ensure images are organized in class folders

**2. CUDA out of memory**
- Reduce `batch_size` in data configuration
- Use smaller architecture (`"simple"` instead of `"deep"`)
- Use `precision: "32"` instead of `"16-mixed"`
- Reduce `num_workers` to 0 or 1

**3. Could not auto-detect num_classes**
- Ensure datamodule can be instantiated and setup successfully
- Check that images are properly organized
- Verify image files are readable

**4. Slow data loading**
- Increase `num_workers` (default: 4)
- Use SSD storage if possible
- Reduce image size if images are very large

**5. Poor model performance**
- Try data augmentation (already enabled by default)
- Increase model complexity (`"medium"` or `"deep"`)
- Train for more epochs
- Check class balance in dataset
- Consider using class weights for imbalanced datasets

**6. Import errors**
- Ensure `PYTHONPATH` is set to project root
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check that you're using the correct Python environment

### Performance Tips

1. **Use GPU**: Set `accelerator: "gpu"` in trainer config
2. **Mixed Precision**: Use `precision: "16-mixed"` for faster training
3. **Batch Size**: Increase `batch_size` if you have GPU memory
4. **Num Workers**: Increase `num_workers` for faster data loading (don't exceed CPU cores)
5. **Image Size**: Smaller images train faster (e.g., `[128, 128]` instead of `[224, 224]`)

## Comparison with Tabular Classification

| Aspect | Tabular Classification | Image Classification |
|--------|----------------------|---------------------|
| **Input** | CSV with features | Images (JPG, PNG, etc.) |
| **Data Format** | CSV file | Folder structure or CSV |
| **Preprocessing** | Scikit-learn (scaling, encoding) | Torchvision (resize, normalize) |
| **Model** | Feedforward MLP | Convolutional Neural Network (CNN) |
| **Architecture** | Hidden layers config | Conv layers + FC layers |
| **Augmentation** | N/A | Built-in (rotation, flip, color jitter) |
| **Input Dimension** | Auto-calculated from features | Fixed by image size |

## Next Steps

- **Transfer Learning**: For better performance, consider using pretrained models (ResNet, EfficientNet, etc.)
- **Data Augmentation**: Experiment with different augmentation strategies
- **Architecture Tuning**: Try custom architectures for your specific task
- **Hyperparameter Tuning**: Use tools like Optuna for automated hyperparameter search
- **Production Deployment**: Create FastAPI web interface similar to StressLevelPrediction API

## License

Same as main project.

## Support

For issues, questions, or contributions, please refer to the main project repository.

