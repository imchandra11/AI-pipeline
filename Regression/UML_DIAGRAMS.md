# Regression Module - UML Diagrams & Architecture Documentation

This document provides comprehensive UML diagrams and architectural documentation for the Regression module, a reusable PyTorch Lightning pipeline for tabular regression tasks.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Class Diagrams](#class-diagrams)
3. [Sequence Diagrams](#sequence-diagrams)
4. [Component Diagrams](#component-diagrams)
5. [Activity Diagrams](#activity-diagrams)
6. [Class-Level Details](#class-level-details)

---

## Architecture Overview

The Regression module is a generic, config-driven pipeline for training regression models on tabular data. It follows the PyTorch Lightning framework and implements a modular architecture with clear separation of concerns:

- **Data Layer**: Handles data loading, preprocessing, and splits
- **Model Layer**: Defines neural network architecture
- **Training Layer**: Manages training loop, optimization, and metrics
- **CLI Layer**: Provides command-line interface for execution
- **Export Layer**: Handles model export for production

---

## Class Diagrams

### Complete Class Diagram

```mermaid
classDiagram
    class RGSLightningCLI {
        -config: LightningConfig
        -model: ModelModuleRGS
        -datamodule: DataModuleRGS
        -trainer: Trainer
        +add_arguments_to_parser(parser)
        +before_instantiate_classes()
        +after_instantiate_classes()
        +link_input_dim()
    }
    
    class DataModuleRGS {
        -csv_path: Path
        -batch_size: int
        -num_workers: int
        -val_split: float
        -test_split: Optional[float]
        -random_seed: int
        -categorical_cols: list[str]
        -numeric_cols: list[str]
        -target_col: str
        -preprocessor: ColumnTransformer
        -input_dim: int
        -train_df: DataFrame
        -val_df: DataFrame
        -test_df: DataFrame
        +setup(stage: str)
        +train_dataloader() DataLoader
        +val_dataloader() DataLoader
        +test_dataloader() DataLoader
        +get_input_dim() int
        -_create_preprocessor()
        -_save_preprocessor()
    }
    
    class RegressionDataset {
        -data: DataFrame
        -preprocessor: ColumnTransformer
        -target_col: str
        -feature_cols: list[str]
        -features: DataFrame
        -targets: ndarray
        -transformed_features: ndarray
        -features_tensor: Tensor
        -targets_tensor: Tensor
        +__len__() int
        +__getitem__(idx) tuple[Tensor, Tensor]
    }
    
    class ModelModuleRGS {
        -model: RegressionModel
        -criterion: MSELoss
        -lr: float
        -weight_decay: float
        -lr_scheduler_factor: Optional[float]
        -lr_scheduler_patience: Optional[int]
        -save_dir: Optional[str]
        -name: Optional[str]
        -training_step_outputs: list
        -validation_step_outputs: list
        +forward(x: Tensor) Tensor
        +training_step(batch, batch_idx) Tensor
        +validation_step(batch, batch_idx) Tensor
        +test_step(batch, batch_idx) dict
        +on_training_epoch_end()
        +on_validation_epoch_end()
        +configure_optimizers() Optimizer
        +configure_schedulers() LRScheduler
    }
    
    class RegressionModel {
        -input_dim: int
        -output_dim: int
        -hidden_layers: list[int]
        -dropout_rates: list[float]
        -activation: str
        -model: nn.Sequential
        +forward(x: Tensor) Tensor
        +get_input_dim() int
        +set_input_dim(dim: int)
        -_get_activation(name: str) nn.Module
        -_build_model()
    }
    
    class ONNXExportCallback {
        -output_dir: Path
        -model_name: str
        -input_dim: Optional[int]
        +on_train_end(trainer, pl_module)
        -_determine_input_dim(trainer, pl_module) int
        -_export_to_onnx(model, input_dim)
    }
    
    %% Relationships
    RGSLightningCLI --> DataModuleRGS : creates & configures
    RGSLightningCLI --> ModelModuleRGS : creates & configures
    RGSLightningCLI --> Trainer : creates
    
    DataModuleRGS --> RegressionDataset : creates instances
    DataModuleRGS ..> sklearn.preprocessing : uses StandardScaler, OneHotEncoder
    DataModuleRGS ..> sklearn.compose : uses ColumnTransformer
    DataModuleRGS ..> joblib : saves/loads preprocessor
    
    RegressionDataset --> torch.utils.data.Dataset : extends
    RegressionDataset --> pandas : uses DataFrame
    RegressionDataset --> numpy : uses arrays
    
    ModelModuleRGS --> RegressionModel : contains
    ModelModuleRGS --> lightning.LightningModule : extends
    ModelModuleRGS --> torch.nn : uses MSELoss
    ModelModuleRGS --> torchmetrics : uses metrics
    
    RegressionModel --> torch.nn.Module : extends
    RegressionModel --> torch.nn : uses Linear, Dropout, etc.
    
    Trainer --> ONNXExportCallback : uses via callbacks
    Trainer --> ModelModuleRGS : trains
    Trainer --> DataModuleRGS : uses for data
```

### Inheritance Hierarchy

```mermaid
classDiagram
    class LightningModule {
        <<abstract>>
        +training_step()
        +validation_step()
        +configure_optimizers()
    }
    
    class LightningDataModule {
        <<abstract>>
        +setup()
        +train_dataloader()
        +val_dataloader()
    }
    
    class LightningCLI {
        <<abstract>>
        +add_arguments_to_parser()
        +before_instantiate_classes()
    }
    
    class Dataset {
        <<abstract>>
        +__len__()
        +__getitem__()
    }
    
    class nn.Module {
        <<abstract>>
        +forward()
    }
    
    class Callback {
        <<abstract>>
        +on_train_end()
    }
    
    ModelModuleRGS --|> LightningModule
    DataModuleRGS --|> LightningDataModule
    RGSLightningCLI --|> LightningCLI
    RegressionDataset --|> Dataset
    RegressionModel --|> nn.Module
    ONNXExportCallback --|> Callback
```

---

## Sequence Diagrams

### Training Workflow Sequence

```mermaid
sequenceDiagram
    participant User
    participant CLI as RGSLightningCLI
    participant Config as config.yaml
    participant DM as DataModuleRGS
    participant Dataset as RegressionDataset
    participant ModelModule as ModelModuleRGS
    participant NNModel as RegressionModel
    participant Trainer
    participant Callback as ONNXExportCallback
    participant Files as File System

    User->>CLI: python main.py --config config.yaml
    CLI->>Config: Load configuration
    Config-->>CLI: Configuration dict
    
    Note over CLI: Phase 1: Configuration & Setup
    CLI->>CLI: before_instantiate_classes()
    CLI->>DM: Create DataModuleRGS(csv_path, ...)
    CLI->>DM: setup('fit')
    DM->>Files: Read CSV file
    Files-->>DM: DataFrame
    DM->>DM: Split train/val/test
    DM->>DM: _create_preprocessor()
    DM->>DM: Fit preprocessor on train data
    DM->>DM: Calculate input_dim
    DM->>Files: Save preprocessor.joblib
    DM-->>CLI: input_dim calculated
    
    CLI->>CLI: Auto-link input_dim to model config
    CLI->>NNModel: Create RegressionModel(input_dim, ...)
    NNModel->>NNModel: _build_model()
    CLI->>ModelModule: Create ModelModuleRGS(model, ...)
    CLI->>Trainer: Create Trainer(callbacks=[...])
    CLI->>Callback: Register ONNXExportCallback
    
    Note over CLI: Phase 2: Training Loop
    CLI->>Trainer: fit(model, datamodule)
    
    loop For each epoch
        Trainer->>DM: train_dataloader()
        DM->>Dataset: Create RegressionDataset(train_df)
        Dataset->>Dataset: Transform features
        Dataset-->>DM: DataLoader(train)
        DM-->>Trainer: DataLoader
        
        loop For each batch
            Trainer->>ModelModule: training_step(batch, batch_idx)
            ModelModule->>NNModel: forward(features)
            NNModel->>NNModel: Forward through layers
            NNModel-->>ModelModule: predictions
            ModelModule->>ModelModule: criterion(predictions, targets)
            ModelModule->>ModelModule: Calculate MSE loss
            ModelModule-->>Trainer: loss
            Trainer->>ModelModule: backward()
            Trainer->>ModelModule: optimizer_step()
        end
        
        Trainer->>DM: val_dataloader()
        DM-->>Trainer: DataLoader(val)
        
        loop For each validation batch
            Trainer->>ModelModule: validation_step(batch, batch_idx)
            ModelModule->>ModelModule: Calculate val metrics
            ModelModule-->>Trainer: val_loss
        end
        
        ModelModule->>ModelModule: on_training_epoch_end()
        ModelModule->>ModelModule: on_validation_epoch_end()
    end
    
    Note over CLI: Phase 3: Model Export
    Trainer->>Callback: on_train_end(trainer, pl_module)
    Callback->>DM: get_input_dim()
    DM-->>Callback: input_dim
    Callback->>NNModel: Export to ONNX
    Callback->>Files: Save model.onnx
    Callback-->>User: ✓ Model exported
```

### Data Flow Sequence

```mermaid
sequenceDiagram
    participant CSV as CSV File
    participant DM as DataModuleRGS
    participant Preprocessor as ColumnTransformer
    participant Dataset as RegressionDataset
    participant DataLoader as DataLoader
    participant Model as ModelModuleRGS

    CSV->>DM: Load data
    DM->>DM: Read CSV to DataFrame
    DM->>DM: Split train/val/test
    
    DM->>Preprocessor: Create ColumnTransformer
    Note over Preprocessor: Categorical: OneHotEncoder<br/>Numeric: StandardScaler
    DM->>Preprocessor: fit(train_features)
    Preprocessor-->>DM: Fitted preprocessor
    
    DM->>Dataset: Create RegressionDataset(df, preprocessor)
    Dataset->>Dataset: Extract features & targets
    Dataset->>Preprocessor: transform(features)
    Preprocessor-->>Dataset: Transformed features (numpy)
    Dataset->>Dataset: Convert to tensors
    
    DM->>DataLoader: DataLoader(dataset, batch_size)
    DataLoader->>Dataset: __getitem__(indices)
    Dataset-->>DataLoader: (features_tensor, targets_tensor)
    
    loop Training Loop
        DataLoader->>Model: Batch of (X, y)
        Model->>Model: Forward pass
        Model->>Model: Loss calculation
    end
```

---

## Component Diagrams

### System Architecture

```mermaid
graph TB
    subgraph "Entry Points"
        Main1[main.py<br/>Standard CLI]
        Main2[mainfittest.py<br/>Fit+Test Workflow]
    end
    
    subgraph "CLI Layer"
        CLI[RGSLightningCLI<br/>Custom Lightning CLI]
        ConfigParser[Config Parser<br/>YAML Handler]
    end
    
    subgraph "Configuration"
        YAML[config.yaml<br/>Hyperparameters<br/>Paths<br/>Model Config]
    end
    
    subgraph "Data Layer"
        DM[DataModuleRGS<br/>Lightning DataModule]
        Dataset[RegressionDataset<br/>PyTorch Dataset]
        Preprocessor[ColumnTransformer<br/>sklearn Pipeline]
        CSV[CSV Data File]
        PreprocessorFile[preprocessor.joblib]
    end
    
    subgraph "Model Layer"
        ModelModule[ModelModuleRGS<br/>Lightning Module]
        NNModel[RegressionModel<br/>Feedforward NN]
        Optimizer[Optimizer<br/>Adam/SGD/RMSprop]
        Scheduler[LR Scheduler<br/>ReduceLROnPlateau]
    end
    
    subgraph "Training Layer"
        Trainer[PyTorch Lightning Trainer]
        Metrics[Training Metrics<br/>MSE, MAE, RMSE]
        Logger[Logger<br/>TensorBoard/CSV]
        Checkpoint[Model Checkpoint<br/>Best/Last]
    end
    
    subgraph "Export Layer"
        Callback[ONNXExportCallback]
        ONNX[model.onnx<br/>Production Model]
    end
    
    subgraph "External Libraries"
        Lightning[PyTorch Lightning]
        PyTorch[PyTorch]
        sklearn[scikit-learn]
        torchmetrics[TorchMetrics]
    end
    
    %% Connections
    Main1 --> CLI
    Main2 --> CLI
    CLI --> ConfigParser
    YAML --> ConfigParser
    ConfigParser --> CLI
    
    CLI --> DM
    CLI --> ModelModule
    CLI --> Trainer
    
    DM --> CSV
    DM --> Preprocessor
    DM --> Dataset
    DM --> PreprocessorFile
    
    Preprocessor --> sklearn
    
    Dataset --> PyTorch
    
    ModelModule --> NNModel
    ModelModule --> Optimizer
    ModelModule --> Scheduler
    ModelModule --> Lightning
    
    NNModel --> PyTorch
    
    Trainer --> ModelModule
    Trainer --> DM
    Trainer --> Metrics
    Trainer --> Logger
    Trainer --> Checkpoint
    Trainer --> Callback
    
    Metrics --> torchmetrics
    
    Callback --> ONNX
    
    style CLI fill:#667eea,color:#fff
    style ModelModule fill:#764ba2,color:#fff
    style Trainer fill:#f093fb,color:#fff
```

### Data Processing Pipeline

```mermaid
graph LR
    A[Raw CSV Data] --> B[Load DataFrame]
    B --> C{Identify Columns}
    C --> D[Categorical Columns]
    C --> E[Numeric Columns]
    C --> F[Target Column]
    
    D --> G[OneHotEncoder]
    E --> H[StandardScaler]
    
    G --> I[ColumnTransformer]
    H --> I
    F --> J[Extract Targets]
    
    I --> K[Fit on Train Data]
    K --> L[Transform All Splits]
    
    L --> M[Train Features]
    L --> N[Val Features]
    L --> O[Test Features]
    
    J --> P[Train Targets]
    J --> Q[Val Targets]
    J --> R[Test Targets]
    
    M --> S[RegressionDataset Train]
    P --> S
    N --> T[RegressionDataset Val]
    Q --> T
    O --> U[RegressionDataset Test]
    R --> U
    
    S --> V[DataLoader Train]
    T --> W[DataLoader Val]
    U --> X[DataLoader Test]
    
    V --> Y[Training Loop]
    W --> Z[Validation Loop]
    X --> AA[Testing Loop]
```

---

## Activity Diagrams

### Training Process Flow

```mermaid
flowchart TD
    Start([Start Training]) --> LoadConfig[Load config.yaml]
    LoadConfig --> ParseConfig[Parse Configuration]
    ParseConfig --> CreateDM[Create DataModuleRGS]
    CreateDM --> SetupDM[Setup DataModule]
    SetupDM --> LoadCSV[Load CSV Data]
    LoadCSV --> SplitData{Split Data?}
    SplitData -->|Yes| TrainValTest[Split: Train/Val/Test]
    SplitData -->|No| TrainVal[Split: Train/Val]
    
    TrainValTest --> CreatePreprocessor[Create Preprocessor]
    TrainVal --> CreatePreprocessor
    CreatePreprocessor --> FitPreprocessor[Fit Preprocessor on Train]
    FitPreprocessor --> CalcInputDim[Calculate input_dim]
    CalcInputDim --> SavePreprocessor{Save Preprocessor?}
    SavePreprocessor -->|Yes| SavePrep[Save to .joblib]
    SavePreprocessor -->|No| CreateModel
    SavePrep --> CreateModel[Create RegressionModel]
    
    CreateModel --> LinkInputDim[Auto-link input_dim]
    LinkInputDim --> CreateModule[Create ModelModuleRGS]
    CreateModule --> CreateTrainer[Create Trainer]
    CreateTrainer --> AddCallbacks[Add Callbacks]
    
    AddCallbacks --> StartTraining[Start Training Loop]
    StartTraining --> Epoch{More Epochs?}
    
    Epoch -->|Yes| TrainBatch[Process Training Batch]
    TrainBatch --> ForwardPass[Forward Pass]
    ForwardPass --> CalcLoss[Calculate MSE Loss]
    CalcLoss --> Backward[Backward Pass]
    Backward --> UpdateWeights[Update Weights]
    UpdateWeights --> ValBatch{Validation?}
    
    ValBatch -->|Yes| ValForward[Validation Forward]
    ValForward --> ValMetrics[Calculate Val Metrics]
    ValMetrics --> LogMetrics[Log Metrics]
    LogMetrics --> Checkpoint{Save Checkpoint?}
    
    ValBatch -->|No| Checkpoint
    Checkpoint -->|Yes| SaveCheckpoint[Save Model]
    Checkpoint -->|No| Epoch
    SaveCheckpoint --> Epoch
    
    Epoch -->|No| ExportONNX[Export to ONNX]
    ExportONNX --> End([Training Complete])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style ExportONNX fill:#87CEEB
```

### Data Module Setup Flow

```mermaid
flowchart TD
    Start([DataModule.setup called]) --> Validate[Validate Configuration]
    Validate --> CheckSplit{Has Test Split?}
    
    CheckSplit -->|Yes| Split3Way[Split: Train/Val/Test]
    CheckSplit -->|No| Split2Way[Split: Train/Val]
    
    Split3Way --> CreatePreprocessor[Create ColumnTransformer]
    Split2Way --> CreatePreprocessor
    
    CreatePreprocessor --> ConfigCategorical{Has Categorical?}
    ConfigCategorical -->|Yes| AddOneHot[Add OneHotEncoder]
    ConfigCategorical -->|No| SkipCategorical
    
    AddOneHot --> ConfigNumeric{Has Numeric?}
    SkipCategorical --> ConfigNumeric
    
    ConfigNumeric -->|Yes| AddScaler[Add StandardScaler]
    ConfigNumeric -->|No| SkipNumeric
    
    AddScaler --> FitPreprocessor[Fit Preprocessor on Train]
    SkipNumeric --> FitPreprocessor
    
    FitPreprocessor --> TransformTrain[Transform Train Data]
    TransformTrain --> TransformVal[Transform Val Data]
    TransformVal --> TransformTest{Has Test?}
    
    TransformTest -->|Yes| TransformTestData[Transform Test Data]
    TransformTest -->|No| CalcDim
    
    TransformTestData --> CalcDim[Calculate input_dim]
    CalcDim --> SavePrep{Save Preprocessor?}
    SavePrep -->|Yes| SaveFile[Save to .joblib]
    SavePrep -->|No| End
    
    SaveFile --> End([Setup Complete])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style FitPreprocessor fill:#87CEEB
```

---

## Class-Level Details

### RGSLightningCLI

**Purpose**: Custom CLI that extends LightningCLI to add regression-specific functionality.

**Key Responsibilities**:
- Parse YAML configuration files
- Instantiate DataModule, Model, and Trainer
- Auto-link `input_dim` from DataModule to Model
- Handle checkpoint paths for resume/testing

**Key Methods**:
- `before_instantiate_classes()`: Auto-detects `input_dim` from DataModule
- `after_instantiate_classes()`: Fallback method to set `input_dim` if needed
- `add_arguments_to_parser()`: Adds custom CLI arguments

---

### DataModuleRGS

**Purpose**: Manages data loading, preprocessing, and splits for regression tasks.

**Key Responsibilities**:
- Load CSV data
- Create and fit preprocessing pipeline
- Split data into train/val/test
- Calculate input dimensions
- Save/load preprocessor for inference

**Key Attributes**:
- `preprocessor`: sklearn ColumnTransformer (fitted)
- `input_dim`: Calculated feature dimension after preprocessing
- `train_df`, `val_df`, `test_df`: Split dataframes

**Key Methods**:
- `setup(stage)`: Prepares data splits and preprocessor
- `train_dataloader()`: Returns DataLoader for training
- `get_input_dim()`: Returns calculated input dimension

---

### RegressionDataset

**Purpose**: PyTorch Dataset that transforms data for model consumption.

**Key Responsibilities**:
- Store preprocessed features and targets as tensors
- Provide indexed access to samples
- Handle feature transformation

**Key Attributes**:
- `features_tensor`: Preprocessed features as FloatTensor
- `targets_tensor`: Targets as FloatTensor

**Key Methods**:
- `__getitem__(idx)`: Returns (features, target) tuple for index

---

### ModelModuleRGS

**Purpose**: PyTorch Lightning module that defines training/validation/test logic.

**Key Responsibilities**:
- Define training step (forward + loss)
- Define validation/test steps with metrics
- Configure optimizer and learning rate scheduler
- Aggregate metrics across batches

**Key Attributes**:
- `model`: RegressionModel instance
- `criterion`: MSELoss for regression
- `training_step_outputs`: Collects batch outputs

**Key Methods**:
- `training_step()`: Single training batch processing
- `validation_step()`: Single validation batch processing
- `configure_optimizers()`: Returns optimizer and scheduler

---

### RegressionModel

**Purpose**: Flexible feedforward neural network for regression.

**Key Responsibilities**:
- Define neural network architecture
- Support configurable layers, dropout, activations
- Forward pass computation

**Key Attributes**:
- `input_dim`: Number of input features
- `hidden_layers`: List of hidden layer sizes
- `model`: nn.Sequential containing all layers

**Key Methods**:
- `forward(x)`: Forward pass through network
- `_build_model()`: Constructs network architecture
- `set_input_dim(dim)`: Sets input dimension if not known at init

---

### ONNXExportCallback

**Purpose**: Exports trained model to ONNX format for production deployment.

**Key Responsibilities**:
- Detect input dimensions
- Export PyTorch model to ONNX
- Save ONNX file to disk

**Key Methods**:
- `on_train_end()`: Called after training completes
- `_determine_input_dim()`: Gets input_dim from datamodule or model

---

## Data Flow Summary

1. **Configuration** → YAML file defines all hyperparameters
2. **Data Loading** → CSV file loaded into DataFrame
3. **Preprocessing** → Categorical one-hot encoded, numeric standardized
4. **Feature Calculation** → `input_dim` = total transformed features
5. **Model Creation** → Feedforward NN with `input_dim` input, 1 output
6. **Training** → Batches → Forward → Loss → Backward → Update
7. **Export** → PyTorch model → ONNX format

---

## Key Design Patterns

1. **Template Method Pattern**: Lightning framework defines training loop structure
2. **Factory Pattern**: ModelFactory creates model instances from config
3. **Strategy Pattern**: Configurable optimizers, schedulers, activations
4. **Observer Pattern**: Callbacks observe training events
5. **Adapter Pattern**: DataModule adapts raw data to PyTorch format

---

## Dependencies

### Core Dependencies
- `lightning` (PyTorch Lightning)
- `torch` (PyTorch)
- `torchmetrics` (Metrics)
- `sklearn` (Preprocessing)
- `pandas` (Data handling)
- `numpy` (Numerical operations)

### External Tools
- `onnxruntime` (ONNX inference)
- `joblib` (Model serialization)
- `tensorboard` (Logging)

---

*Last Updated: [Current Date]*
