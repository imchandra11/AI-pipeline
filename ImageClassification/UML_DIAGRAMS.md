# Image Classification Module - UML Diagrams & Architecture Documentation

This document provides comprehensive UML diagrams and architectural documentation for the ImageClassification module, a reusable PyTorch Lightning pipeline for image classification tasks using CNNs.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Class Diagrams](#class-diagrams)
3. [Sequence Diagrams](#sequence-diagrams)
4. [Component Diagrams](#component-diagrams)
5. [Activity Diagrams](#activity-diagrams)
6. [Class-Level Details](#class-level-details)

---

## Architecture Overview

The ImageClassification module is a generic, config-driven pipeline for training CNN-based image classification models. It differs significantly from tabular modules:

- **CNN Architecture**: Uses Convolutional Neural Networks instead of feedforward networks
- **Image Data**: Handles image files (JPG, PNG, etc.) instead of CSV tables
- **Image Preprocessing**: Resize, normalize, augment images
- **Data Organization**: Supports folder-based (class folders) or CSV-based (image paths + labels)
- **Transfer Learning Ready**: Architecture designed for easy extension to pretrained models

Key Components:
- **Data Layer**: Image loading, augmentation, transforms
- **Model Layer**: CNN with configurable architecture
- **Training Layer**: Standard PyTorch Lightning training with image-specific optimizations
- **Export Layer**: ONNX export for production deployment

---

## Class Diagrams

### Complete Class Diagram

```mermaid
classDiagram
    class IMGLightningCLI {
        -config: LightningConfig
        -model: ModelModuleIMG
        -datamodule: DataModuleIMG
        -trainer: Trainer
        +add_arguments_to_parser(parser)
        +before_instantiate_classes()
        +after_instantiate_classes()
        +link_num_classes()
    }
    
    class DataModuleIMG {
        -data_dir: Optional[Path]
        -csv_path: Optional[Path]
        -image_path_col: str
        -label_col: str
        -image_size: tuple[int, int]
        -batch_size: int
        -num_workers: int
        -val_split: float
        -test_split: Optional[float]
        -random_seed: int
        -augmentation: dict
        -normalization: dict
        -label_encoder: Optional[LabelEncoder]
        -class_names: list[str]
        -num_classes: int
        -image_paths: list[Path]
        -labels: list
        -train_paths: list[Path]
        -val_paths: list[Path]
        -test_paths: list[Path]
        +setup(stage: str)
        +train_dataloader() DataLoader
        +val_dataloader() DataLoader
        +test_dataloader() DataLoader
        +get_num_classes() int
        -_load_from_folders() tuple
        -_load_from_csv() tuple
        -_create_transforms()
        -_encode_labels()
        -_save_preprocessor()
    }
    
    class ImageClassificationDataset {
        -image_paths: list[Path]
        -labels: list[int]
        -transform: transforms.Compose
        -class_to_idx: dict
        +__len__() int
        +__getitem__(idx) tuple[Tensor, Tensor]
        -_load_image(path: Path) Image
    }
    
    class ModelModuleIMG {
        -model: ImageClassificationModel
        -criterion: CrossEntropyLoss
        -lr: float
        -weight_decay: float
        -lr_scheduler_factor: Optional[float]
        -lr_scheduler_patience: Optional[int]
        -save_dir: Optional[str]
        -name: Optional[str]
        -train_accuracy: MulticlassAccuracy
        -val_accuracy: MulticlassAccuracy
        -test_accuracy: MulticlassAccuracy
        -train_f1: MulticlassF1Score
        -val_f1: MulticlassF1Score
        -test_f1: MulticlassF1Score
        -train_precision: MulticlassPrecision
        -val_precision: MulticlassPrecision
        -test_precision: MulticlassPrecision
        -train_recall: MulticlassRecall
        -val_recall: MulticlassRecall
        -test_recall: MulticlassRecall
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
    
    class ImageClassificationModel {
        -input_channels: int
        -num_classes: int
        -architecture: str
        -input_size: tuple[int, int]
        -activation: str
        -conv_layers: nn.Module
        -fc_layers: nn.Module
        -model: nn.Sequential
        +forward(x: Tensor) Tensor
        +get_num_classes() int
        +get_input_channels() int
        +set_num_classes(classes: int)
        -_build_preset_conv_layers(arch: str, in_channels: int) nn.Module
        -_build_custom_conv_layers(configs: list, in_channels: int) nn.Module
        -_build_fc_layers(hidden_dims: list, dropout_rates: list)
        -_calculate_feature_size() int
        -_get_activation(name: str) nn.Module
    }
    
    class ONNXExportCallback {
        -output_dir: Path
        -model_name: str
        -input_shape: Optional[list[int]]
        +on_train_end(trainer, pl_module)
        -_determine_input_shape(trainer, pl_module) list[int]
        -_export_to_onnx(model, input_shape)
    }
    
    %% Relationships
    IMGLightningCLI --> DataModuleIMG : creates & configures
    IMGLightningCLI --> ModelModuleIMG : creates & configures
    IMGLightningCLI --> Trainer : creates
    
    DataModuleIMG --> ImageClassificationDataset : creates instances
    DataModuleIMG ..> PIL.Image : loads images
    DataModuleIMG ..> torchvision.transforms : uses for preprocessing
    DataModuleIMG ..> sklearn.preprocessing : uses LabelEncoder
    DataModuleIMG ..> joblib : saves/loads label_encoder
    DataModuleIMG ..> pathlib : handles file paths
    
    ImageClassificationDataset --> torch.utils.data.Dataset : extends
    ImageClassificationDataset --> PIL.Image : loads & processes images
    ImageClassificationDataset --> torchvision.transforms : applies transforms
    
    ModelModuleIMG --> ImageClassificationModel : contains
    ModelModuleIMG --> lightning.LightningModule : extends
    ModelModuleIMG --> torch.nn : uses CrossEntropyLoss
    ModelModuleIMG --> torchmetrics : uses Accuracy, F1Score, Precision, Recall
    
    ImageClassificationModel --> torch.nn.Module : extends
    ImageClassificationModel --> torch.nn : uses Conv2d, MaxPool2d, Linear, etc.
    
    Trainer --> ONNXExportCallback : uses via callbacks
    Trainer --> ModelModuleIMG : trains
    Trainer --> DataModuleIMG : uses for data
```

### CNN Architecture Components

```mermaid
classDiagram
    class ImageClassificationModel {
        <<CNN Model>>
        -conv_layers: nn.Sequential
        -fc_layers: nn.Sequential
        +forward(x)
    }
    
    class ConvBlock {
        -conv: Conv2d
        -bn: BatchNorm2d
        -activation: ReLU/Tanh/etc
        -pool: MaxPool2d
        +forward(x)
    }
    
    class FCBlock {
        -linear: Linear
        -dropout: Dropout
        -activation: ReLU/Tanh/etc
        +forward(x)
    }
    
    ImageClassificationModel --> ConvBlock : contains multiple
    ImageClassificationModel --> FCBlock : contains multiple
    
    note for ImageClassificationModel "Architecture Presets:\n- Simple: 2 conv blocks\n- Medium: 4 conv blocks\n- Deep: 6 conv blocks\n- Custom: User-defined"
```

---

## Sequence Diagrams

### Training Workflow Sequence

```mermaid
sequenceDiagram
    participant User
    participant CLI as IMGLightningCLI
    participant Config as config.yaml
    participant DM as DataModuleIMG
    participant Dataset as ImageClassificationDataset
    participant ModelModule as ModelModuleIMG
    participant CNNModel as ImageClassificationModel
    participant Trainer
    participant Callback as ONNXExportCallback
    participant Files as File System

    User->>CLI: python main.py --config config.yaml
    CLI->>Config: Load configuration
    Config-->>CLI: Configuration dict
    
    Note over CLI: Phase 1: Configuration & Setup
    CLI->>CLI: before_instantiate_classes()
    CLI->>DM: Create DataModuleIMG(data_dir/csv_path, ...)
    CLI->>DM: setup('fit')
    
    alt Folder-based loading
        DM->>Files: Scan class folders
        Files-->>DM: image_paths, labels
    else CSV-based loading
        DM->>Files: Read CSV file
        Files-->>DM: image_paths, labels
    end
    
    DM->>DM: Extract unique classes
    DM->>DM: _encode_labels()
    DM->>DM: Split train/val/test
    DM->>DM: _create_transforms()
    Note over DM: Create train transforms<br/>(with augmentation)<br/>Create val/test transforms<br/>(no augmentation)
    DM->>DM: Calculate num_classes
    DM->>Files: Save label_encoder.joblib
    DM-->>CLI: num_classes calculated
    
    CLI->>CLI: Auto-link num_classes to model config
    CLI->>CNNModel: Create ImageClassificationModel(input_channels, num_classes, ...)
    CNNModel->>CNNModel: _build_preset_conv_layers() or _build_custom_conv_layers()
    CNNModel->>CNNModel: _calculate_feature_size()
    CNNModel->>CNNModel: _build_fc_layers()
    CLI->>ModelModule: Create ModelModuleIMG(model, ...)
    ModelModule->>ModelModule: Initialize metrics (Accuracy, F1, Precision, Recall)
    CLI->>Trainer: Create Trainer(callbacks=[...])
    CLI->>Callback: Register ONNXExportCallback
    
    Note over CLI: Phase 2: Training Loop
    CLI->>Trainer: fit(model, datamodule)
    
    loop For each epoch
        Trainer->>DM: train_dataloader()
        DM->>Dataset: Create ImageClassificationDataset(train_paths, train_labels)
        Dataset-->>DM: DataLoader(train)
        DM-->>Trainer: DataLoader
        
        loop For each batch
            Trainer->>Dataset: __getitem__(indices)
            Dataset->>Dataset: _load_image(path)
            Dataset->>Dataset: Apply train_transform
            Dataset-->>Trainer: (image_tensor, label_tensor)
            
            Trainer->>ModelModule: training_step(batch, batch_idx)
            ModelModule->>CNNModel: forward(images)
            CNNModel->>CNNModel: Forward through conv layers
            CNNModel->>CNNModel: Flatten feature maps
            CNNModel->>CNNModel: Forward through FC layers
            CNNModel-->>ModelModule: logits [batch_size, num_classes]
            ModelModule->>ModelModule: criterion(logits, labels)
            ModelModule->>ModelModule: Calculate CrossEntropy loss
            ModelModule->>ModelModule: Update accuracy metrics
            ModelModule->>ModelModule: Update F1, precision, recall
            ModelModule-->>Trainer: loss
            Trainer->>ModelModule: backward()
            Trainer->>ModelModule: optimizer_step()
        end
        
        Trainer->>DM: val_dataloader()
        DM-->>Trainer: DataLoader(val)
        
        loop For each validation batch
            Trainer->>Dataset: __getitem__(indices)
            Dataset->>Dataset: Apply val_transform (no augmentation)
            Trainer->>ModelModule: validation_step(batch, batch_idx)
            ModelModule->>ModelModule: Calculate val metrics
            ModelModule-->>Trainer: val_loss, val_accuracy, val_f1
        end
        
        ModelModule->>ModelModule: on_training_epoch_end()
        ModelModule->>ModelModule: on_validation_epoch_end()
    end
    
    Note over CLI: Phase 3: Model Export
    Trainer->>Callback: on_train_end(trainer, pl_module)
    Callback->>DM: Get image_size
    Callback->>CNNModel: get_input_channels()
    Callback->>Callback: Determine input_shape [C, H, W]
    Callback->>CNNModel: Export to ONNX
    Callback->>Files: Save model.onnx
    Callback-->>User: ✓ Model exported
```

### Image Loading & Preprocessing Sequence

```mermaid
sequenceDiagram
    participant DM as DataModuleIMG
    participant Dataset as ImageClassificationDataset
    participant PIL as PIL.Image
    participant Transforms as torchvision.transforms
    participant Model as CNN

    DM->>DM: Load image paths & labels
    DM->>DM: Create train/val/test transforms
    
    Note over DM,Transforms: Train Transforms:<br/>Resize, Augment, ToTensor, Normalize
    Note over DM,Transforms: Val/Test Transforms:<br/>Resize, ToTensor, Normalize
    
    DM->>Dataset: Create Dataset(paths, labels, transform)
    
    loop For each batch
        Dataset->>Dataset: __getitem__(idx)
        Dataset->>PIL: Image.open(path)
        PIL-->>Dataset: PIL Image object
        Dataset->>Dataset: Convert to RGB if needed
        Dataset->>Transforms: transform(image)
        
        alt Training
            Transforms->>Transforms: Resize(image_size)
            Transforms->>Transforms: RandomRotation(angle)
            Transforms->>Transforms: RandomHorizontalFlip()
            Transforms->>Transforms: ColorJitter()
            Transforms->>Transforms: ToTensor()
            Transforms->>Transforms: Normalize(mean, std)
        else Validation/Test
            Transforms->>Transforms: Resize(image_size)
            Transforms->>Transforms: ToTensor()
            Transforms->>Transforms: Normalize(mean, std)
        end
        
        Transforms-->>Dataset: image_tensor [C, H, W]
        Dataset-->>Model: Batch of images [B, C, H, W]
        Model->>Model: Forward through CNN
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
        CLI[IMGLightningCLI<br/>Custom Lightning CLI]
        ConfigParser[Config Parser<br/>YAML Handler]
    end
    
    subgraph "Configuration"
        YAML[config.yaml<br/>Hyperparameters<br/>Paths<br/>Model Config<br/>Augmentation]
    end
    
    subgraph "Data Layer"
        DM[DataModuleIMG<br/>Lightning DataModule]
        Dataset[ImageClassificationDataset<br/>PyTorch Dataset]
        DataLoader[DataLoader<br/>Batch Loading]
        Transforms[Image Transforms<br/>torchvision]
        Augmentation[Data Augmentation<br/>Rotation, Flip, Jitter]
        Normalization[Image Normalization<br/>Mean/Std Scaling]
        LabelEncoder[LabelEncoder<br/>Class Encoding]
        
        subgraph "Data Sources"
            FolderData[Folder Structure<br/>class1/, class2/, ...]
            CSVData[CSV with Paths<br/>image_path, label]
        end
        
        LabelEncoderFile[label_encoder.joblib]
    end
    
    subgraph "Model Layer"
        ModelModule[ModelModuleIMG<br/>Lightning Module]
        CNNModel[ImageClassificationModel<br/>CNN Architecture]
        
        subgraph "CNN Components"
            ConvLayers[Convolutional Layers<br/>Conv2d, BatchNorm, ReLU]
            PoolingLayers[Pooling Layers<br/>MaxPool2d]
            FCLayers[Fully Connected Layers<br/>Linear, Dropout]
        end
        
        Optimizer[Optimizer<br/>Adam/SGD/RMSprop]
        Scheduler[LR Scheduler<br/>ReduceLROnPlateau]
    end
    
    subgraph "Training Layer"
        Trainer[PyTorch Lightning Trainer]
        Metrics[Classification Metrics<br/>Accuracy, F1, Precision, Recall]
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
        torchvision[torchvision]
        sklearn[scikit-learn]
        PIL[Pillow/PIL]
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
    
    DM --> FolderData
    DM --> CSVData
    DM --> Dataset
    DM --> Transforms
    DM --> LabelEncoder
    DM --> LabelEncoderFile
    
    Dataset --> DataLoader
    Dataset --> PIL
    Dataset --> Transforms
    
    Transforms --> Augmentation
    Transforms --> Normalization
    
    Augmentation --> torchvision
    Normalization --> torchvision
    
    LabelEncoder --> sklearn
    
    ModelModule --> CNNModel
    ModelModule --> Optimizer
    ModelModule --> Scheduler
    ModelModule --> Lightning
    
    CNNModel --> ConvLayers
    CNNModel --> PoolingLayers
    CNNModel --> FCLayers
    
    ConvLayers --> PyTorch
    FCLayers --> PyTorch
    
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
    style CNNModel fill:#ff6b6b,color:#fff
    style ConvLayers fill:#4ecdc4,color:#fff
```

### CNN Architecture Flow

```mermaid
graph LR
    A[Input Image<br/>B, C, H, W] --> B[Conv2d Layer]
    B --> C[BatchNorm2d]
    C --> D[Activation<br/>ReLU/Tanh/etc]
    D --> E{Has Pooling?}
    E -->|Yes| F[MaxPool2d]
    E -->|No| G[Next Conv Block]
    F --> G
    
    G --> H{More Conv Blocks?}
    H -->|Yes| B
    H -->|No| I[Flatten<br/>B, C*H*W]
    
    I --> J[FC Layer 1]
    J --> K[Dropout]
    K --> L[Activation]
    L --> M{More FC Layers?}
    M -->|Yes| J
    M -->|No| N[Output Layer<br/>B, num_classes]
    
    N --> O[Logits]
    
    style A fill:#90EE90
    style O fill:#FFB6C1
    style B fill:#87CEEB
    style F fill:#DDA0DD
    style J fill:#F0E68C
```

---

## Activity Diagrams

### Image Classification Training Process

```mermaid
flowchart TD
    Start([Start Training]) --> LoadConfig[Load config.yaml]
    LoadConfig --> ParseConfig[Parse Configuration]
    ParseConfig --> CreateDM[Create DataModuleIMG]
    CreateDM --> SetupDM[Setup DataModule]
    
    SetupDM --> CheckData{Data Source?}
    CheckData -->|Folder| LoadFolders[Load from class folders<br/>Scan directory structure]
    CheckData -->|CSV| LoadCSV[Load from CSV<br/>Read image_path, label]
    
    LoadFolders --> ExtractClasses[Extract class names<br/>from folder names]
    LoadCSV --> ExtractClassesCSV[Extract class names<br/>from CSV labels]
    
    ExtractClasses --> EncodeLabels
    ExtractClassesCSV --> EncodeLabels[Encode labels with LabelEncoder]
    
    EncodeLabels --> GetClasses[Get num_classes & class_names]
    GetClasses --> SplitData{Split Data?}
    SplitData -->|Yes| TrainValTest[Split: Train/Val/Test]
    SplitData -->|No| TrainVal[Split: Train/Val]
    
    TrainValTest --> CreateTransforms[Create Image Transforms]
    TrainVal --> CreateTransforms
    
    CreateTransforms --> ConfigTrain[Train Transforms:<br/>Resize + Augmentation<br/>+ Normalize]
    ConfigTrain --> ConfigVal[Val/Test Transforms:<br/>Resize + Normalize<br/>No Augmentation]
    
    ConfigVal --> SaveLabelEnc{Save Label Encoder?}
    SaveLabelEnc -->|Yes| SaveLE[Save label_encoder.joblib]
    SaveLabelEnc -->|No| CreateModel
    
    SaveLE --> CreateModel[Create ImageClassificationModel]
    
    CreateModel --> CheckArch{Architecture?}
    CheckArch -->|Preset| BuildPreset[Build Preset Architecture<br/>simple/medium/deep]
    CheckArch -->|Custom| BuildCustom[Build Custom Architecture<br/>from config]
    
    BuildPreset --> CalcFeatureSize[Calculate Feature Map Size<br/>After Conv Layers]
    BuildCustom --> CalcFeatureSize
    
    CalcFeatureSize --> BuildFC[Build FC Layers]
    BuildFC --> LinkClasses[Auto-link num_classes]
    LinkClasses --> CreateModule[Create ModelModuleIMG]
    CreateModule --> InitMetrics[Initialize Metrics<br/>Accuracy, F1, Precision, Recall]
    InitMetrics --> CreateTrainer[Create Trainer]
    CreateTrainer --> AddCallbacks[Add Callbacks]
    
    AddCallbacks --> StartTraining[Start Training Loop]
    StartTraining --> Epoch{More Epochs?}
    
    Epoch -->|Yes| LoadBatch[Load Image Batch]
    LoadBatch --> LoadImages[Load Images from Disk<br/>PIL.Image.open]
    LoadImages --> ApplyTrainTransform[Apply Train Transform<br/>with Augmentation]
    ApplyTrainTransform --> ForwardPass[Forward Pass through CNN]
    
    ForwardPass --> ConvForward[Conv Layers Forward]
    ConvForward --> Flatten[Flatten Feature Maps]
    Flatten --> FCForward[FC Layers Forward]
    FCForward --> CalcLoss[Calculate CrossEntropy Loss]
    
    CalcLoss --> UpdateMetrics[Update Classification Metrics]
    UpdateMetrics --> Backward[Backward Pass]
    Backward --> UpdateWeights[Update Weights]
    UpdateWeights --> ValBatch{Validation?}
    
    ValBatch -->|Yes| LoadValImages[Load Val Images]
    LoadValImages --> ApplyValTransform[Apply Val Transform<br/>No Augmentation]
    ApplyValTransform --> ValForward[Validation Forward]
    ValForward --> ValMetrics[Calculate Val Metrics]
    ValMetrics --> LogMetrics[Log Metrics]
    LogMetrics --> Checkpoint{Save Checkpoint?}
    
    ValBatch -->|No| Checkpoint
    Checkpoint -->|Yes| SaveCheckpoint[Save Model]
    Checkpoint -->|No| Epoch
    SaveCheckpoint --> Epoch
    
    Epoch -->|No| ExportONNX[Export to ONNX<br/>with Image Input Shape]
    ExportONNX --> End([Training Complete])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style ExportONNX fill:#87CEEB
    style ConvForward fill:#FFD700
    style FCForward fill:#FFD700
    style ApplyTrainTransform fill:#DDA0DD
```

### Image Preprocessing Pipeline

```mermaid
flowchart LR
    A[Raw Image File] --> B[PIL.Image.open]
    B --> C{Image Mode?}
    C -->|RGB| D[Use as-is]
    C -->|Grayscale| E[Convert to RGB]
    C -->|Other| E
    
    D --> F[Resize to image_size]
    E --> F
    
    F --> G{Is Training?}
    G -->|Yes| H[Data Augmentation]
    G -->|No| K[Skip Augmentation]
    
    H --> I[RandomRotation]
    I --> J[RandomHorizontalFlip]
    J --> L[ColorJitter]
    L --> M[RandomCrop]
    M --> K
    
    K --> N[ToTensor<br/>Convert to Tensor]
    N --> O[Normalize<br/>Mean/Std Scaling]
    O --> P[Final Image Tensor<br/>C, H, W]
    
    style A fill:#90EE90
    style P fill:#FFB6C1
    style H fill:#FFD700
    style N fill:#87CEEB
```

---

## Class-Level Details

### IMGLightningCLI

**Purpose**: Custom CLI for image classification with auto-linking of `num_classes`.

**Key Responsibilities**:
- Parse YAML configuration
- Instantiate DataModule, Model, and Trainer
- Auto-link `num_classes` from DataModule to Model
- Handle image-specific configurations

**Key Methods**:
- `before_instantiate_classes()`: Auto-detects `num_classes` from DataModule
- `after_instantiate_classes()`: Fallback to set `num_classes` if needed

**Difference from Tabular Modules**: Links only `num_classes` (no `input_dim` needed - fixed by image size).

---

### DataModuleIMG

**Purpose**: Manages image data loading, preprocessing, and augmentation.

**Key Responsibilities**:
- Load images from folder structure or CSV
- Create train/val/test splits
- Define image transforms (with/without augmentation)
- Encode labels and save label encoder
- Auto-detect number of classes

**Key Attributes**:
- `data_dir`: Directory with class folders (optional)
- `csv_path`: CSV file with image paths (optional)
- `image_size`: Target image size (height, width)
- `augmentation`: Dictionary with augmentation settings
- `normalization`: Dictionary with normalization mean/std
- `label_encoder`: sklearn LabelEncoder for class names
- `num_classes`: Number of unique classes

**Key Methods**:
- `_load_from_folders()`: Loads images from class folders
- `_load_from_csv()`: Loads images from CSV file
- `_create_transforms()`: Creates train/val/test transforms
- `get_num_classes()`: Returns number of classes

**Data Organization**:
1. **Folder-based**: `data/images/class1/img1.jpg`, `data/images/class2/img2.jpg`
2. **CSV-based**: CSV with columns `image_path`, `label`

---

### ImageClassificationDataset

**Purpose**: PyTorch Dataset that loads and transforms images on-the-fly.

**Key Responsibilities**:
- Load images from disk (lazy loading)
- Apply transforms (resize, augment, normalize)
- Return (image_tensor, label_tensor) tuples

**Key Attributes**:
- `image_paths`: List of image file paths
- `labels`: List of encoded labels (integers)
- `transform`: torchvision.Compose transform pipeline
- `class_to_idx`: Dictionary mapping class names to indices

**Key Methods**:
- `_load_image(path)`: Loads and converts image to RGB
- `__getitem__(idx)`: Returns (image_tensor, label_tensor)

**Memory Efficiency**: Images loaded on-demand (not all in memory).

---

### ModelModuleIMG

**Purpose**: PyTorch Lightning module for image classification.

**Similar to ModelModuleCLS**: Uses same metrics and loss, but for images.

**Key Differences**:
- Works with image tensors [B, C, H, W] instead of feature vectors
- CNN model processes spatial data

---

### ImageClassificationModel

**Purpose**: Flexible CNN architecture for image classification.

**Key Responsibilities**:
- Define CNN architecture (conv + FC layers)
- Support preset architectures (simple, medium, deep)
- Support custom architectures
- Calculate feature map sizes automatically

**Key Attributes**:
- `input_channels`: 1 (grayscale) or 3 (RGB)
- `num_classes`: Number of output classes
- `architecture`: Preset name or 'custom'
- `conv_layers`: Sequential of conv blocks
- `fc_layers`: Sequential of fully connected layers

**Architecture Presets**:
1. **Simple**: 2 conv blocks (32, 64 channels)
2. **Medium**: 4 conv blocks (64, 128, 256, 512 channels)
3. **Deep**: 6 conv blocks (64, 128, 256, 256, 512, 512 channels)

**Key Methods**:
- `_build_preset_conv_layers()`: Builds preset architecture
- `_build_custom_conv_layers()`: Builds custom architecture
- `_calculate_feature_size()`: Calculates FC input size after conv layers
- `get_input_channels()`: Returns number of input channels

**Feature Size Calculation**: Automatically calculates feature map dimensions after all conv+pooling layers to determine FC input size.

---

### ONNXExportCallback

**Purpose**: Exports CNN model to ONNX with image input shape.

**Key Differences from Tabular**:
- Input shape is `[C, H, W]` (not `[features]`)
- Needs to determine image channels, height, width
- Example: `[3, 224, 224]` for RGB 224x224 images

**Key Methods**:
- `_determine_input_shape()`: Gets [C, H, W] from datamodule and model
- `_export_to_onnx()`: Exports with image input shape

---

## Key Design Patterns

1. **Template Method Pattern**: Lightning framework defines training loop
2. **Factory Pattern**: ModelFactory creates CNN architectures
3. **Strategy Pattern**: Configurable architectures, optimizers, augmentation
4. **Lazy Loading Pattern**: Images loaded on-demand in Dataset
5. **Adapter Pattern**: Transforms adapt raw images to tensor format

---

## Critical Implementation Details

### Image Preprocessing Pipeline

**Training Transforms**:
1. Resize to `image_size` (e.g., 224x224)
2. **Augmentation** (randomly applied):
   - Rotation (configurable angle)
   - Horizontal flip
   - Vertical flip (optional)
   - Color jitter (brightness, contrast, saturation, hue)
   - Random crop (optional)
3. Convert to tensor (0-1 range)
4. Normalize with mean/std (ImageNet defaults or custom)

**Validation/Test Transforms**:
1. Resize to `image_size`
2. Convert to tensor
3. Normalize with mean/std
4. **No augmentation** (deterministic)

### CNN Feature Size Calculation

The module automatically calculates feature map dimensions after conv layers:

1. Start with input size: `[H, W]` (e.g., 224x224)
2. For each conv block:
   - Apply conv (may change channels)
   - Apply pooling (reduces H, W by pooling factor)
3. Final feature size: `C × H' × W'`
4. FC input = flattened: `C × H' × W'`

This allows automatic FC layer sizing without manual calculation.

### Label Encoding

Similar to Classification module:
- Encodes string labels to integers
- Remaps non-0-indexed integers to 0-indexed
- Stores mapping for inverse transform (prediction → class name)

---

## Comparison: Tabular vs Image Modules

| Aspect | Regression/Classification | ImageClassification |
|--------|-------------------------|---------------------|
| **Input** | Tabular features (CSV) | Images (files) |
| **Model** | Feedforward NN | CNN |
| **Preprocessing** | sklearn (scaling, encoding) | torchvision (resize, normalize) |
| **Data Loading** | DataFrame → Tensor | Image files → Tensor |
| **Augmentation** | N/A | Rotation, flip, color jitter |
| **Input Shape** | `[batch, features]` | `[batch, channels, height, width]` |
| **Feature Extraction** | Direct from CSV | Learned by conv layers |

---

*Last Updated: [Current Date]*
