# AI Pipeline - Complete UML Documentation Overview

This document provides a high-level overview of the entire AI pipeline architecture, showing how the three reusable modules (Regression, Classification, ImageClassification) work together and share common design patterns.

## Table of Contents

1. [System Overview](#system-overview)
2. [Module Comparison](#module-comparison)
3. [Unified Architecture](#unified-architecture)
4. [Common Design Patterns](#common-design-patterns)
5. [Module Relationships](#module-relationships)
6. [Workflow Comparison](#workflow-comparison)

---

## System Overview

The AI pipeline consists of three independent, reusable modules that follow a consistent architecture pattern but are specialized for different tasks:

```
AI-Pipeline/
├── Regression/          # Tabular regression (e.g., price prediction)
├── Classification/      # Tabular classification (e.g., stress level)
└── ImageClassification/ # Image classification (e.g., food recognition)
```

All modules share:
- **PyTorch Lightning** framework
- **Config-driven** approach (YAML)
- **Modular architecture** (DataModule, ModelModule, ModelFactory)
- **ONNX export** for production
- **Auto-detection** of dimensions/classes

---

## Module Comparison

### High-Level Comparison

```mermaid
graph TB
    subgraph "Regression Module"
        R1[Tabular Data]
        R2[Feedforward NN]
        R3[MSELoss]
        R4[1 Output]
        R5[MSE, MAE, RMSE]
        R1 --> R2 --> R3 --> R4 --> R5
    end
    
    subgraph "Classification Module"
        C1[Tabular Data]
        C2[Feedforward NN]
        C3[CrossEntropyLoss]
        C4[N Classes Output]
        C5[Accuracy, F1, Precision, Recall]
        C1 --> C2 --> C3 --> C4 --> C5
    end
    
    subgraph "ImageClassification Module"
        I1[Image Files]
        I2[CNN]
        I3[CrossEntropyLoss]
        I4[N Classes Output]
        I5[Accuracy, F1, Precision, Recall]
        I1 --> I2 --> I3 --> I4 --> I5
    end
    
    style R2 fill:#667eea,color:#fff
    style C2 fill:#764ba2,color:#fff
    style I2 fill:#ff6b6b,color:#fff
```

### Detailed Feature Comparison

| Feature | Regression | Classification | ImageClassification |
|---------|-----------|---------------|---------------------|
| **Input Type** | CSV (tabular) | CSV (tabular) | Images (files) |
| **Data Organization** | CSV columns | CSV columns | Folder structure or CSV |
| **Model Type** | Feedforward NN | Feedforward NN | CNN |
| **Input Format** | Feature vector | Feature vector | Image tensor [C,H,W] |
| **Preprocessing** | sklearn (scaling, encoding) | sklearn (scaling, encoding) | torchvision (resize, augment) |
| **Output** | 1 continuous value | N class logits | N class logits |
| **Loss Function** | MSELoss | CrossEntropyLoss | CrossEntropyLoss |
| **Metrics** | MSE, MAE, RMSE | Accuracy, F1, Precision, Recall | Accuracy, F1, Precision, Recall |
| **Auto-Detection** | input_dim | input_dim, num_classes | num_classes |
| **Label Encoding** | N/A | Yes (to 0-indexed) | Yes (to 0-indexed) |
| **Augmentation** | N/A | N/A | Yes (rotation, flip, etc.) |

---

## Unified Architecture

### Common Architecture Pattern

```mermaid
graph TB
    subgraph "All Modules Share This Pattern"
        A[config.yaml] --> B[Custom CLI]
        B --> C[DataModule]
        B --> D[ModelModule]
        B --> E[Trainer]
        C --> F[Dataset]
        D --> G[Model]
        E --> D
        E --> C
        E --> H[ONNXExportCallback]
        H --> I[model.onnx]
    end
    
    style B fill:#667eea,color:#fff
    style C fill:#764ba2,color:#fff
    style D fill:#f093fb,color:#fff
    style G fill:#ff6b6b,color:#fff
```

### Shared Component Structure

```mermaid
classDiagram
    class BaseCLI {
        <<abstract>>
        +before_instantiate_classes()
        +after_instantiate_classes()
        +add_arguments_to_parser()
    }
    
    class BaseDataModule {
        <<abstract>>
        +setup()
        +train_dataloader()
        +val_dataloader()
        +test_dataloader()
    }
    
    class BaseModelModule {
        <<abstract>>
        +training_step()
        +validation_step()
        +configure_optimizers()
    }
    
    class BaseModel {
        <<abstract>>
        +forward()
    }
    
    RGSLightningCLI --|> BaseCLI
    CLSLightningCLI --|> BaseCLI
    IMGLightningCLI --|> BaseCLI
    
    DataModuleRGS --|> BaseDataModule
    DataModuleCLS --|> BaseDataModule
    DataModuleIMG --|> BaseDataModule
    
    ModelModuleRGS --|> BaseModelModule
    ModelModuleCLS --|> BaseModelModule
    ModelModuleIMG --|> BaseModelModule
    
    RegressionModel --|> BaseModel
    ClassificationModel --|> BaseModel
    ImageClassificationModel --|> BaseModel
```

---

## Common Design Patterns

### 1. Template Method Pattern

All modules use PyTorch Lightning's template method pattern:

```mermaid
sequenceDiagram
    participant Lightning
    participant Module as ModelModule
    participant User
    
    Lightning->>Module: training_step(batch)
    Module->>Module: forward(x)
    Module->>Module: criterion(pred, target)
    Module-->>Lightning: loss
    Lightning->>Lightning: backward()
    Lightning->>Lightning: optimizer_step()
    
    Note over Lightning,Module: Lightning defines the structure<br/>Modules implement the steps
```

### 2. Factory Pattern

Model factories create model instances from configuration:

```mermaid
graph LR
    A[config.yaml] --> B[Model Config]
    B --> C[ModelFactory]
    C --> D[RegressionModel]
    C --> E[ClassificationModel]
    C --> F[ImageClassificationModel]
    
    style C fill:#667eea,color:#fff
```

### 3. Strategy Pattern

Configurable components use strategy pattern:

- **Optimizers**: Adam, SGD, RMSprop (configurable)
- **Schedulers**: ReduceLROnPlateau, OneCycleLR (configurable)
- **Activations**: ReLU, Tanh, GELU (configurable)
- **Architectures**: Simple, Medium, Deep (ImageClassification)

### 4. Observer Pattern

Callbacks observe training events:

```mermaid
graph TB
    A[Trainer] --> B[ONNXExportCallback]
    A --> C[ModelCheckpoint]
    A --> D[TensorBoardLogger]
    
    B --> E[on_train_end]
    C --> F[on_validation_end]
    D --> G[on_batch_end]
```

---

## Module Relationships

### Shared Dependencies

```mermaid
graph TB
    subgraph "Core Framework"
        L[PyTorch Lightning]
        T[PyTorch]
        TM[TorchMetrics]
    end
    
    subgraph "Data Processing"
        SK[scikit-learn]
        PD[pandas]
        NP[numpy]
        TV[torchvision]
        PIL[Pillow]
    end
    
    subgraph "Export & Serialization"
        ONNX[ONNX Runtime]
        JB[joblib]
    end
    
    subgraph "Modules"
        REG[Regression]
        CLS[Classification]
        IMG[ImageClassification]
    end
    
    REG --> L
    REG --> T
    REG --> SK
    REG --> PD
    
    CLS --> L
    CLS --> T
    CLS --> SK
    CLS --> PD
    CLS --> TM
    
    IMG --> L
    IMG --> T
    IMG --> TV
    IMG --> PIL
    IMG --> TM
    
    REG --> ONNX
    CLS --> ONNX
    IMG --> ONNX
    
    REG --> JB
    CLS --> JB
    IMG --> JB
    
    style L fill:#667eea,color:#fff
    style T fill:#764ba2,color:#fff
```

---

## Workflow Comparison

### Regression Workflow

```mermaid
flowchart LR
    A[CSV Data] --> B[Preprocess]
    B --> C[Train NN]
    C --> D[Predict Price]
    
    style C fill:#667eea,color:#fff
```

### Classification Workflow

```mermaid
flowchart LR
    A[CSV Data] --> B[Preprocess]
    B --> C[Encode Labels]
    C --> D[Train NN]
    D --> E[Predict Class]
    
    style C fill:#764ba2,color:#fff
    style D fill:#764ba2,color:#fff
```

### Image Classification Workflow

```mermaid
flowchart LR
    A[Image Files] --> B[Load & Augment]
    B --> C[Train CNN]
    C --> D[Predict Class]
    
    style B fill:#ff6b6b,color:#fff
    style C fill:#ff6b6b,color:#fff
```

---

## Unified Training Flow

All modules follow the same high-level training flow:

```mermaid
flowchart TD
    Start([Start]) --> Config[Load config.yaml]
    Config --> Create[Create Components]
    Create --> Setup[Setup Data]
    Setup --> Link[Auto-link Dimensions]
    Link --> Train[Train Model]
    Train --> Validate[Validate]
    Validate --> Export[Export ONNX]
    Export --> End([End])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style Train fill:#87CEEB
    style Export fill:#FFD700
```

---

## Module-Specific Details

### Regression Module
- **Focus**: Continuous value prediction
- **Key Feature**: Auto-detects `input_dim`
- **Use Case**: Price prediction, regression tasks
- **Documentation**: See `Regression/UML_DIAGRAMS.md`

### Classification Module
- **Focus**: Multi-class classification
- **Key Features**: 
  - Auto-detects `input_dim` and `num_classes`
  - Ensures 0-indexed labels for CrossEntropyLoss
- **Use Case**: Stress level prediction, categorical classification
- **Documentation**: See `Classification/UML_DIAGRAMS.md`

### Image Classification Module
- **Focus**: Image-based classification
- **Key Features**:
  - Supports folder-based or CSV-based data organization
  - Image augmentation for training
  - CNN architectures (simple, medium, deep, custom)
  - Auto-calculates feature map sizes
- **Use Case**: Food classification, image recognition
- **Documentation**: See `ImageClassification/UML_DIAGRAMS.md`

---

## How to Use This Documentation

1. **Start Here**: Read this overview to understand the overall architecture
2. **Module Details**: Dive into specific module documentation:
   - `Regression/UML_DIAGRAMS.md` - For regression tasks
   - `Classification/UML_DIAGRAMS.md` - For classification tasks
   - `ImageClassification/UML_DIAGRAMS.md` - For image classification
3. **Implementation**: Use the class diagrams and sequence diagrams to understand implementation details
4. **Extension**: Use the architecture patterns to extend or create new modules

---

## Key Takeaways

1. **Consistency**: All modules follow the same architectural pattern
2. **Modularity**: Each module is independent and reusable
3. **Config-Driven**: Everything configured via YAML files
4. **Auto-Detection**: Automatic dimension/class detection reduces errors
5. **Production-Ready**: ONNX export for all modules
6. **Extensibility**: Easy to add new modules following the same pattern

---

## Next Steps

- Explore individual module documentation for detailed UML diagrams
- Review example configurations in project directories
- See README files for usage instructions
- Check example projects (GemstonePricePrediction, StressLevelPrediction, FoodClassification)

---

*Last Updated: [Current Date]*
*For detailed module documentation, see individual UML_DIAGRAMS.md files in each module directory.*
