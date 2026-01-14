# AI Model Training Pipeline - Regression Module

A generic, modular, and reusable PyTorch Lightning pipeline for training regression models. This pipeline is fully config-driven, allowing you to train models on any tabular regression dataset by simply modifying a YAML configuration file.

## Features

- **Fully Config-Driven**: All settings (features, hyperparameters, paths) controlled via YAML files
- **Generic & Reusable**: Use the same codebase for any regression task (gemstone prices, house prices, etc.)
- **Auto-Dimension Detection**: Automatically calculates input dimensions from feature lists
- **Production-Ready**: Exports models to ONNX format with preprocessors for easy deployment
- **PyTorch Lightning**: Built on PyTorch Lightning for scalable, professional ML training

## Project Structure

```
AI-pipeline/
├── Regression/                    # Generic regression module (reusable)
│   ├── __init__.py
│   ├── cli.py                    # Custom Lightning CLI
│   ├── main.py                   # Standard CLI entry point
│   ├── mainfittest.py            # Fit+test workflow entry point
│   ├── dataset.py                # PyTorch Dataset for tabular data
│   ├── datamodule.py             # Lightning DataModule
│   ├── modelfactory.py           # Neural network model factory
│   ├── modelmodule.py            # Lightning Module for training
│   └── callbacks.py              # ONNX export callback
├── GemstonePricePriction/         # Example project (config-only)
│   ├── configs/
│   │   ├── gemstone.yaml         # Main configuration
│   │   └── gemstone.local.yaml   # Local overrides
│   ├── data/
│   │   └── gemstone.csv          # Dataset
│   ├── models/                   # Output: trained models & preprocessors
│   └── lightning_logs/           # Output: training logs
├── requirements.txt              # Python dependencies
├── venv/                         # Virtual environment (create this)
└── README.md                     # This file
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

### 3. Verify Installation

```bash
python -c "import lightning as L; import torch; print('Installation successful!')"
```

## Creating a New Project

To train a model on a new dataset, follow these steps:

### Step 1: Create Project Directory Structure

```bash
mkdir YourProjectName
mkdir YourProjectName/configs
mkdir YourProjectName/data
mkdir YourProjectName/models
mkdir YourProjectName/lightning_logs
```

### Step 2: Place Your Dataset

Place your CSV file in `YourProjectName/data/your_data.csv`

**CSV Requirements:**
- Must contain feature columns (categorical and/or numeric)
- Must contain a target column (the value to predict)
- No missing values in target column

### Step 3: Create Configuration File

Create `YourProjectName/configs/your_project.yaml`:

```yaml
# Your Project Configuration
seed_everything: true

trainer:
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        filename: "{epoch}-{val_loss:.2f}.best"
        monitor: "val_loss"
        mode: "min"
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
    - class_path: Regression.callbacks.ONNXExportCallback
      init_args:
        output_dir: "models"
        model_name: "your_model_name"
        input_dim: null  # Auto-detected

  logger:
    class_path: lightning.pytorch.loggers.TensorBoardLogger
    init_args:
      save_dir: "lightning_logs"
      name: "YourProjectTraining"
      default_hp_metric: false

  max_epochs: &me 50
  num_sanity_val_steps: 1
  check_val_every_n_epoch: 1
  log_every_n_steps: 10
  accelerator: auto
  devices: auto
  precision: 16-mixed
  default_root_dir: "lightning_logs/YourProjectTraining"

model:
  class_path: Regression.modelmodule.ModelModuleRGS
  init_args:
    lr: 0.0001
    weight_decay: 0.0
    lr_scheduler_factor: 0.5
    lr_scheduler_patience: 5
    save_dir: "models"
    name: "your_model_name"
    model:
      class_path: Regression.modelfactory.RegressionModel
      init_args:
        input_dim: 0  # Auto-set from datamodule
        hidden_layers: [128, 64, 32]
        dropout_rates: [0.15, 0.1, 0.05]
        activation: "relu"

optimizer: 
  class_path: torch.optim.Adam
  init_args:
    lr: 0.001
    weight_decay: 0.00001

lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: 0.001
    pct_start: 0.1
    total_steps: *me

data:
  class_path: Regression.datamodule.DataModuleRGS
  init_args:
    csv_path: "YourProjectName/data/your_data.csv"
    batch_size: 256
    num_workers: 0
    val_split: 0.2
    random_seed: 42
    categorical_cols:
      - column1
      - column2
      # Add your categorical column names here
    numeric_cols:
      - column3
      - column4
      # Add your numeric column names here
    target_col: "target"  # Your target column name
    save_preprocessor: true
    preprocessor_path: "YourProjectName/models/preprocessor.joblib"

fit:
  ckpt_path: null   # Set to checkpoint path for resume training

test:
  ckpt_path: best   # Use "best" or "last" checkpoint
```

### Step 4: Create Local Override File (Optional)

Create `YourProjectName/configs/your_project.local.yaml` for local-specific settings:

```yaml
# Local configuration overrides
# This file is optional and can override settings from main config
# Example:
# trainer:
#   max_epochs: 10
#   precision: 32
# data:
#   init_args:
#     batch_size: 128
```

## Running the Project

**Important:** Always run commands from the project root directory (where `Regression/` folder is located).

### Option 1: Fit + Test Workflow (Recommended for Quick Testing)

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "."; python Regression/mainfittest.py --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

**Windows (CMD):**
```cmd
set PYTHONPATH=. && python Regression/mainfittest.py --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

**Linux/Mac:**
```bash
PYTHONPATH=. python Regression/mainfittest.py \
  --config YourProjectName/configs/your_project.yaml \
  --config YourProjectName/configs/your_project.local.yaml
```

### Option 2: Standard Lightning CLI Commands

**Training only (Windows PowerShell):**
```powershell
$env:PYTHONPATH = "."; python Regression/main.py fit --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

**Testing only (Windows PowerShell):**
```powershell
$env:PYTHONPATH = "."; python Regression/main.py test --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

**Resume training from checkpoint:**
```powershell
$env:PYTHONPATH = "."; python Regression/main.py fit --config YourProjectName/configs/your_project.yaml --fit.ckpt_path "YourProjectName/lightning_logs/YourProjectTraining/version_0/checkpoints/epoch-10.last.ckpt"
```

**Linux/Mac:**
```bash
PYTHONPATH=. python Regression/main.py fit \
  --config YourProjectName/configs/your_project.yaml \
  --config YourProjectName/configs/your_project.local.yaml
```

### Option 3: Using Helper Scripts (Easier)

**Windows (PowerShell):**
```powershell
.\run_training.ps1 --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

**Linux/Mac:**
```bash
chmod +x run_training.sh
./run_training.sh --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

### Option 4: Using Jupyter Notebook

Create a notebook (e.g., `YourProjectTrainer.ipynb`):

```python
import sys
import os
# Add project root to path
sys.path.insert(0, os.getcwd())

# Import regression modules
from Regression.modelmodule import ModelModuleRGS
from Regression.datamodule import DataModuleRGS

# Run fit and test workflow
# Note: You may need to set PYTHONPATH in the notebook environment
%run Regression/mainfittest.py --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

## Configuration Guide

### Data Configuration

**Key Parameters:**
- `csv_path`: Path to your CSV file (relative to project root)
- `batch_size`: Batch size for training (default: 256)
- `val_split`: Validation split ratio (0.0 to 1.0, default: 0.2)
- `categorical_cols`: List of categorical feature column names
- `numeric_cols`: List of numeric feature column names
- `target_col`: Name of the target column to predict
- `preprocessor_path`: Where to save/load the preprocessor

**Preprocessing:**
- Categorical columns: Automatically one-hot encoded (with `drop='first'`)
- Numeric columns: Automatically standardized using StandardScaler
- Input dimension: Automatically calculated from feature lists

### Model Configuration

**Key Parameters:**
- `hidden_layers`: List of hidden layer sizes, e.g., `[128, 64, 32]`
- `dropout_rates`: List of dropout rates matching hidden layers, e.g., `[0.15, 0.1, 0.05]`
- `activation`: Activation function (`"relu"`, `"tanh"`, `"gelu"`, `"sigmoid"`, `"leaky_relu"`, `"elu"`)
- `input_dim`: Automatically set from datamodule (set to `0` in config)

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
   - `preprocessor.joblib`: Fitted preprocessor for data transformation

2. **Checkpoints** (`YourProjectName/lightning_logs/YourProjectTraining/version_X/checkpoints/`):
   - `epoch-X-val_loss=Y.best.ckpt`: Best model checkpoint
   - `epoch-X.last.ckpt`: Last epoch checkpoint

3. **Training Logs** (`YourProjectName/lightning_logs/`):
   - TensorBoard logs for visualization

## Viewing Training Progress

### TensorBoard

```bash
tensorboard --logdir YourProjectName/lightning_logs
```

Then open `http://localhost:6006` in your browser.

## Example Projects

### Gemstone Price Prediction

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "."; python Regression/mainfittest.py --config GemstonePricePriction/configs/gemstone.yaml --config GemstonePricePriction/configs/gemstone.local.yaml
```

**Or using helper script:**
```powershell
.\run_training.ps1 --config GemstonePricePriction/configs/gemstone.yaml --config GemstonePricePriction/configs/gemstone.local.yaml
```

**Linux/Mac:**
```bash
PYTHONPATH=. python Regression/mainfittest.py \
  --config GemstonePricePriction/configs/gemstone.yaml \
  --config GemstonePricePriction/configs/gemstone.local.yaml
```

**Dataset:** `GemstonePricePriction/data/gemstone.csv`
- **Categorical features:** cut, color, clarity
- **Numeric features:** carat, depth, table, x, y, z
- **Target:** price

### House Price Prediction (Example)

To create a house price prediction project:

1. Create `HousePricePrediction/` directory structure
2. Place your house data CSV in `HousePricePrediction/data/houses.csv`
3. Create config file with:
   - Categorical columns: `neighborhood`, `house_type`, `condition`
   - Numeric columns: `sqft`, `bedrooms`, `bathrooms`, `year_built`
   - Target: `price`
4. Run training as shown above

## Troubleshooting

### Common Issues

**1. FileNotFoundError: CSV file not found**
- Check that `csv_path` in config is correct relative to project root
- Ensure CSV file exists at the specified path

**2. ValueError: Missing columns in CSV**
- Verify all column names in `categorical_cols` and `numeric_cols` exist in CSV
- Check for typos in column names

**3. Could not auto-detect input_dim**
- Ensure datamodule can be instantiated and setup successfully
- Check that CSV file is readable and has valid data

**4. CUDA out of memory**
- Reduce `batch_size` in data configuration
- Reduce model size (smaller `hidden_layers`)
- Use `precision: "32"` instead of `"16-mixed"`

**5. Import errors**
- Ensure virtual environment is activated
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Set `PYTHONPATH=.` before running (or use helper scripts)

**6. DLL loading errors on Windows (OSError: [WinError 1114])**
- This is often a PyTorch/Windows compatibility issue
- Try reinstalling PyTorch: `pip uninstall torch && pip install torch`
- Ensure you have Visual C++ Redistributables installed
- Try using CPU-only version: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Advanced Usage

### Custom Model Architecture

Modify `hidden_layers` and `dropout_rates` in config:

```yaml
model:
  init_args:
    model:
      init_args:
        hidden_layers: [256, 128, 64, 32]  # Deeper network
        dropout_rates: [0.2, 0.15, 0.1, 0.05]
        activation: "gelu"
```

### Hyperparameter Tuning

Override hyperparameters in local config:

```yaml
# your_project.local.yaml
model:
  init_args:
    lr: 0.0005
optimizer:
  init_args:
    lr: 0.002
    weight_decay: 0.0001
data:
  init_args:
    batch_size: 512
```

### Resume Training

```bash
python Regression/main.py fit \
  --config YourProjectName/configs/your_project.yaml \
  --fit.ckpt_path "lightning_logs/YourProjectTraining/version_0/checkpoints/epoch-10.last.ckpt"
```

## Model Inference

After training, use the exported ONNX model and preprocessor:

```python
import joblib
import onnxruntime as ort
import numpy as np
import pandas as pd

# Load preprocessor
preprocessor = joblib.load("YourProjectName/models/preprocessor.joblib")

# Load ONNX model
session = ort.InferenceSession("YourProjectName/models/your_model_name.onnx")

# Prepare input data
input_data = pd.DataFrame({
    'categorical_col': ['value1'],
    'numeric_col': [123.45],
    # ... other features
})

# Transform data
transformed = preprocessor.transform(input_data[feature_cols])

# Predict
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: transformed.astype(np.float32)})
prediction = output[0][0]

print(f"Prediction: {prediction}")
```

## Contributing

This is a generic pipeline designed to be extended. To add new features:

1. Modify files in `Regression/` for generic improvements
2. Keep project-specific code in project directories (configs only)
3. Follow the config-driven approach

## License

[Add your license here]

## Support

For issues or questions, please check:
1. Configuration file syntax
2. CSV file format and column names
3. Dependencies installation
4. TensorBoard logs for training insights

