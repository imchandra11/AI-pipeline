# Food Classification API - UML Diagrams & Architecture Documentation

This document provides comprehensive UML diagrams and architectural documentation for the Food Classification API, showing how the ONNX model is integrated with FastAPI and how the web interface connects to the inference pipeline.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Class Diagrams](#class-diagrams)
3. [Sequence Diagrams](#sequence-diagrams)
4. [Component Diagrams](#component-diagrams)
5. [Activity Diagrams](#activity-diagrams)
6. [Class-Level Details](#class-level-details)

---

## Architecture Overview

The Food Classification API is a production-ready inference service that serves a trained CNN model via FastAPI. It provides both a web interface and REST API endpoints for food image classification.

**Key Components**:
- **FastAPI Application**: HTTP server with web UI and REST API
- **ONNX Runtime**: Efficient model inference engine
- **Predictor Service**: Handles image preprocessing and ONNX inference
- **Web Interface**: Jinja2-templated HTML with drag-and-drop upload
- **Model Artifacts**: ONNX model and label encoder (loaded at startup)

**Architecture Layers**:
1. **Presentation Layer**: Web UI (HTML/CSS/JS) and REST API
2. **Application Layer**: FastAPI routes and request handling
3. **Service Layer**: Predictor class (preprocessing + inference)
4. **Model Layer**: ONNX Runtime with loaded model
5. **Data Layer**: Model files (ONNX, label encoder)

---

## Class Diagrams

### Complete Class Diagram

```mermaid
classDiagram
    class FastAPI {
        +app: FastAPI
        +lifespan: asynccontextmanager
        +mount(static_files)
        +get(path)
        +post(path)
    }
    
    class FoodClassificationApp {
        -app: FastAPI
        -predictor: Optional[FoodClassificationPredictor]
        -templates: Jinja2Templates
        -MAX_UPLOAD_SIZE: int
        -ALLOWED_EXTENSIONS: set
        +lifespan(app: FastAPI)
        +validate_image(file: UploadFile) bytes
        +home(request: Request) HTMLResponse
        +predict_web(request, image: UploadFile) HTMLResponse
        +predict_api(image: UploadFile) FoodPrediction
        +get_classes() dict
        +health_check() dict
    }
    
    class FoodClassificationPredictor {
        -model_path: Path
        -label_encoder_path: Path
        -image_size: tuple[int, int]
        -normalization_mean: list[float]
        -normalization_std: list[float]
        -session: Optional[InferenceSession]
        -label_encoder: Optional[LabelEncoder]
        -class_names: Optional[list[str]]
        -transform: Optional[transforms.Compose]
        +__init__(model_path, label_encoder_path, ...)
        +load()
        +preprocess_image(image: Image.Image) np.ndarray
        +predict(image: Image.Image, top_k: Optional[int]) dict
        +predict_from_bytes(image_bytes: bytes, top_k: Optional[int]) dict
        +get_class_names() list[str]
    }
    
    class FoodPrediction {
        +predicted_class: str
        +confidence: float
        +top_predictions: List[FoodPredictionItem]
        +all_probabilities: Dict[str, float]
        +num_classes: int
    }
    
    class FoodPredictionItem {
        +class_name: str
        +confidence: float
        +class_index: int
    }
    
    class Jinja2Templates {
        +TemplateResponse(template, context) Response
    }
    
    class InferenceSession {
        +get_inputs() list
        +run(output_names, input_dict) list
    }
    
    class ImageTransform {
        +Resize(size)
        +ToTensor()
        +Normalize(mean, std)
    }
    
    class LabelEncoder {
        +classes_: ndarray
        +inverse_transform(indices) list
    }
    
    %% Relationships
    FoodClassificationApp --> FastAPI : uses
    FoodClassificationApp --> FoodClassificationPredictor : contains
    FoodClassificationApp --> Jinja2Templates : uses
    FoodClassificationApp --> FoodPrediction : returns
    FoodClassificationApp --> FoodPredictionItem : uses
    
    FoodClassificationPredictor --> InferenceSession : uses
    FoodClassificationPredictor --> ImageTransform : uses
    FoodClassificationPredictor --> LabelEncoder : uses
    FoodClassificationPredictor --> PIL.Image : processes
    
    FoodPrediction --> FoodPredictionItem : contains
    
    InferenceSession ..> ONNXModel : loads
    LabelEncoder ..> joblib : loads from
    
    note for FoodClassificationApp "Endpoints:\n- GET / (web UI)\n- POST /predict (web form)\n- POST /api/predict (REST API)\n- GET /api/classes\n- GET /health"
    
    note for FoodClassificationPredictor "Handles:\n- Model loading\n- Image preprocessing\n- ONNX inference\n- Probability calculation\n- Class name mapping"
```

### Request/Response Flow

```mermaid
classDiagram
    class Client {
        <<Browser/HTTP Client>>
        +send_request()
        +render_html()
    }
    
    class FastAPIRoute {
        <<HTTP Endpoint>>
        +handle_request()
        +validate_input()
        +process_request()
    }
    
    class PredictorService {
        <<Business Logic>>
        +preprocess()
        +infer()
        +postprocess()
    }
    
    class ONNXRuntime {
        <<Inference Engine>>
        +run_session()
        +get_output()
    }
    
    class ModelArtifacts {
        <<File System>>
        +model.onnx
        +label_encoder.joblib
    }
    
    Client --> FastAPIRoute : HTTP Request
    FastAPIRoute --> PredictorService : process()
    PredictorService --> ONNXRuntime : inference()
    ONNXRuntime --> ModelArtifacts : loads
    ONNXRuntime --> PredictorService : logits
    PredictorService --> FastAPIRoute : predictions
    FastAPIRoute --> Client : HTTP Response
```

---

## Sequence Diagrams

### Web Interface Prediction Flow

```mermaid
sequenceDiagram
    participant User
    participant Browser
    participant FastAPI as FastAPI App
    participant Route as /predict Route
    participant Validator as Image Validator
    participant Predictor as FoodClassificationPredictor
    participant ONNX as ONNX Runtime
    participant Model as ONNX Model
    participant LabelEnc as Label Encoder
    participant Template as Jinja2 Template
    participant Response as HTML Response

    User->>Browser: Upload food image (drag & drop)
    Browser->>FastAPI: POST /predict (multipart/form-data)
    
    FastAPI->>Route: predict_web(request, image: UploadFile)
    Route->>Validator: validate_image(file)
    Validator->>Validator: Check file extension
    Validator->>Validator: Check file size (< 10MB)
    Validator->>Validator: Verify image format (PIL)
    Validator-->>Route: image_bytes
    
    Route->>Route: Image.open(io.BytesIO(image_bytes))
    Route->>Predictor: predict(image_obj, top_k=3)
    
    Predictor->>Predictor: preprocess_image(image)
    Predictor->>Predictor: Convert to RGB
    Predictor->>Predictor: Apply transforms (Resize, ToTensor, Normalize)
    Predictor->>Predictor: Add batch dimension [1, 3, 224, 224]
    Predictor-->>Predictor: preprocessed_array
    
    Predictor->>ONNX: session.run(None, {input_name: array})
    ONNX->>Model: Forward pass
    Model-->>ONNX: logits [num_classes]
    ONNX-->>Predictor: logits
    
    Predictor->>Predictor: Apply softmax (numerically stable)
    Predictor->>Predictor: Get top 3 predictions
    Predictor->>LabelEnc: Map indices to class names
    LabelEnc-->>Predictor: class_names
    Predictor->>Predictor: Build prediction dict
    Predictor-->>Route: prediction dict
    
    Route->>Route: Encode image to base64 for display
    Route->>Template: TemplateResponse("index.html", context)
    Template->>Template: Render HTML with prediction
    Template-->>Route: HTML content
    Route->>Response: HTMLResponse(html)
    Response-->>Browser: HTML with results
    Browser->>User: Display prediction results
```

### REST API Prediction Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI as FastAPI App
    participant Route as /api/predict Route
    participant Validator as Image Validator
    participant Predictor as FoodClassificationPredictor
    participant ONNX as ONNX Runtime
    participant Pydantic as FoodPrediction Model
    participant Response as JSON Response

    Client->>FastAPI: POST /api/predict (multipart/form-data)
    
    FastAPI->>Route: predict_api(image: UploadFile)
    Route->>Validator: validate_image(file)
    Validator-->>Route: image_bytes
    
    Route->>Route: Image.open(io.BytesIO(image_bytes))
    Route->>Predictor: predict(image_obj, top_k=3)
    
    Predictor->>Predictor: preprocess_image(image)
    Predictor->>ONNX: session.run(...)
    ONNX-->>Predictor: logits
    Predictor->>Predictor: Calculate probabilities & top K
    Predictor-->>Route: prediction dict
    
    Route->>Pydantic: FoodPrediction(**prediction)
    Pydantic->>Pydantic: Validate & serialize
    Pydantic-->>Route: FoodPrediction object
    Route->>Response: JSONResponse(prediction)
    Response-->>Client: JSON with predictions
```

### Application Startup Sequence

```mermaid
sequenceDiagram
    participant Server
    participant FastAPI as FastAPI App
    participant Lifespan as Lifespan Handler
    participant Predictor as FoodClassificationPredictor
    participant FileSystem as File System
    participant ONNX as ONNX Runtime
    participant Joblib as joblib
    participant LabelEnc as Label Encoder

    Server->>FastAPI: Start application
    FastAPI->>Lifespan: lifespan(app) [startup]
    
    Lifespan->>Lifespan: Setup paths (models directory)
    Lifespan->>Predictor: FoodClassificationPredictor(...)
    Predictor->>Predictor: Store configuration
    
    Lifespan->>Predictor: load()
    Predictor->>FileSystem: Check model.onnx exists
    FileSystem-->>Predictor: model.onnx
    
    Predictor->>ONNX: InferenceSession(model_path)
    ONNX->>FileSystem: Load ONNX model
    FileSystem-->>ONNX: Model bytes
    ONNX->>ONNX: Initialize session
    ONNX-->>Predictor: session object
    
    Predictor->>FileSystem: Check label_encoder.joblib exists
    FileSystem-->>Predictor: label_encoder.joblib
    
    Predictor->>Joblib: load(label_encoder_path)
    Joblib->>FileSystem: Read file
    FileSystem-->>Joblib: Serialized encoder
    Joblib-->>Predictor: LabelEncoder object
    
    Predictor->>LabelEnc: encoder.classes_
    LabelEnc-->>Predictor: class_names list
    
    Predictor->>Predictor: Create transform pipeline
    Predictor->>Predictor: transforms.Compose([Resize, ToTensor, Normalize])
    Predictor-->>Lifespan: Load complete
    
    Lifespan->>Lifespan: Print success message
    Lifespan-->>FastAPI: Application ready
    FastAPI-->>Server: Server running on port 8000
```

### Image Preprocessing Pipeline

```mermaid
sequenceDiagram
    participant Route
    participant PIL as PIL.Image
    participant Predictor as FoodClassificationPredictor
    participant Transform as Image Transform
    participant NumPy as NumPy Array
    participant ONNX as ONNX Runtime

    Route->>PIL: Image.open(io.BytesIO(image_bytes))
    PIL-->>Route: Image object
    
    Route->>Predictor: preprocess_image(image)
    Predictor->>PIL: image.convert('RGB')
    PIL-->>Predictor: RGB Image
    
    Predictor->>Transform: transform(image)
    Transform->>Transform: Resize((224, 224))
    Transform->>Transform: ToTensor() [0-1 range]
    Transform->>Transform: Normalize(mean, std)
    Transform-->>Predictor: image_tensor [3, 224, 224]
    
    Predictor->>NumPy: tensor.numpy()
    NumPy-->>Predictor: numpy array [3, 224, 224]
    
    Predictor->>NumPy: expand_dims(array, axis=0)
    NumPy-->>Predictor: array [1, 3, 224, 224]
    
    Predictor->>NumPy: astype(np.float32)
    NumPy-->>Predictor: float32 array
    
    Predictor->>ONNX: session.run(None, {input_name: array})
    ONNX-->>Predictor: logits [num_classes]
```

---

## Component Diagrams

### System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        Browser[Web Browser<br/>User Interface]
        RESTClient[REST API Client<br/>curl/Postman/etc]
    end
    
    subgraph "HTTP Layer"
        FastAPI[FastAPI Application<br/>HTTP Server]
        Uvicorn[Uvicorn ASGI Server]
    end
    
    subgraph "Route Layer"
        WebRoute[Web Routes<br/>GET /, POST /predict]
        APIRoute[API Routes<br/>POST /api/predict<br/>GET /api/classes<br/>GET /health]
    end
    
    subgraph "Application Layer"
        Validator[Image Validator<br/>File validation]
        Predictor[FoodClassificationPredictor<br/>Inference Service]
        TemplateEngine[Jinja2 Templates<br/>HTML Rendering]
    end
    
    subgraph "Service Layer"
        Preprocessor[Image Preprocessor<br/>Resize, Normalize]
        ONNXRuntime[ONNX Runtime<br/>Inference Engine]
        LabelMapper[Label Mapper<br/>Index to Class Name]
    end
    
    subgraph "Model Layer"
        ONNXModel[ONNX Model<br/>food_classification_model.onnx]
        LabelEncoder[Label Encoder<br/>label_encoder.joblib]
    end
    
    subgraph "Data Layer"
        ModelFiles[Model Files<br/>File System]
        StaticFiles[Static Files<br/>CSS, JS]
        TemplateFiles[Template Files<br/>HTML Templates]
    end
    
    %% Client to HTTP
    Browser --> FastAPI
    RESTClient --> FastAPI
    FastAPI --> Uvicorn
    
    %% HTTP to Routes
    FastAPI --> WebRoute
    FastAPI --> APIRoute
    
    %% Routes to Application
    WebRoute --> Validator
    WebRoute --> Predictor
    WebRoute --> TemplateEngine
    APIRoute --> Validator
    APIRoute --> Predictor
    
    %% Application to Service
    Predictor --> Preprocessor
    Predictor --> ONNXRuntime
    Predictor --> LabelMapper
    
    %% Service to Model
    ONNXRuntime --> ONNXModel
    LabelMapper --> LabelEncoder
    
    %% Model to Data
    ONNXModel --> ModelFiles
    LabelEncoder --> ModelFiles
    
    %% Static/Template Files
    FastAPI --> StaticFiles
    TemplateEngine --> TemplateFiles
    
    style FastAPI fill:#667eea,color:#fff
    style Predictor fill:#764ba2,color:#fff
    style ONNXRuntime fill:#f093fb,color:#fff
    style ONNXModel fill:#ff6b6b,color:#fff
```

### Request Processing Pipeline

```mermaid
graph LR
    A[HTTP Request] --> B{Request Type?}
    
    B -->|Web UI| C[GET /]
    B -->|Web Form| D[POST /predict]
    B -->|REST API| E[POST /api/predict]
    B -->|Info| F[GET /api/classes]
    B -->|Health| G[GET /health]
    
    C --> H[Render Template<br/>with class list]
    H --> I[HTML Response]
    
    D --> J[Validate Image]
    E --> J
    
    J --> K{Valid?}
    K -->|No| L[Error Response]
    K -->|Yes| M[Load Image]
    
    M --> N[Preprocess Image]
    N --> O[ONNX Inference]
    O --> P[Postprocess Results]
    
    P --> Q{Response Type?}
    Q -->|Web| R[Render Template<br/>with results]
    Q -->|API| S[Serialize JSON]
    
    R --> T[HTML Response]
    S --> U[JSON Response]
    
    F --> V[Get Class Names]
    V --> W[JSON Response]
    
    G --> X[Health Status]
    X --> W
    
    style O fill:#ff6b6b,color:#fff
    style P fill:#764ba2,color:#fff
```

---

## Activity Diagrams

### Complete Prediction Workflow

```mermaid
flowchart TD
    Start([User Uploads Image]) --> Receive[Receive HTTP Request]
    Receive --> ValidateFile{Validate File?}
    
    ValidateFile -->|Invalid Extension| Error1[Return Error: Invalid Format]
    ValidateFile -->|File Too Large| Error2[Return Error: File Too Large]
    ValidateFile -->|Invalid Image| Error3[Return Error: Invalid Image]
    ValidateFile -->|Valid| LoadImage[Load Image with PIL]
    
    LoadImage --> ConvertRGB{Is RGB?}
    ConvertRGB -->|No| Convert[Convert to RGB]
    ConvertRGB -->|Yes| Preprocess
    Convert --> Preprocess[Preprocess Image]
    
    Preprocess --> Resize[Resize to 224x224]
    Resize --> ToTensor[Convert to Tensor]
    ToTensor --> Normalize[Normalize with Mean/Std]
    Normalize --> AddBatch[Add Batch Dimension]
    AddBatch --> ONNXInference[ONNX Runtime Inference]
    
    ONNXInference --> GetLogits[Get Logits Array]
    GetLogits --> Softmax[Apply Softmax]
    Softmax --> GetTopK[Get Top K Predictions]
    GetTopK --> MapClasses[Map Indices to Class Names]
    MapClasses --> BuildResponse[Build Response Dict]
    
    BuildResponse --> CheckType{Response Type?}
    CheckType -->|Web UI| EncodeImage[Encode Image to Base64]
    CheckType -->|REST API| SerializeJSON[Serialize to JSON]
    
    EncodeImage --> RenderTemplate[Render Jinja2 Template]
    RenderTemplate --> HTMLResponse[Return HTML Response]
    
    SerializeJSON --> ValidatePydantic[Validate with Pydantic]
    ValidatePydantic --> JSONResponse[Return JSON Response]
    
    HTMLResponse --> End([User Sees Results])
    JSONResponse --> End
    
    Error1 --> End
    Error2 --> End
    Error3 --> End
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style ONNXInference fill:#87CEEB
    style Softmax fill:#FFD700
```

### Application Startup Workflow

```mermaid
flowchart TD
    Start([Server Starts]) --> InitFastAPI[Initialize FastAPI App]
    InitFastAPI --> RegisterLifespan[Register Lifespan Handler]
    RegisterLifespan --> Startup[Lifespan Startup]
    
    Startup --> SetupPaths[Setup Model Paths]
    SetupPaths --> CheckModel{Model File Exists?}
    
    CheckModel -->|No| Error1[Raise FileNotFoundError]
    CheckModel -->|Yes| CheckEncoder{Encoder File Exists?}
    
    CheckEncoder -->|No| Error2[Raise FileNotFoundError]
    CheckEncoder -->|Yes| CreatePredictor[Create Predictor Instance]
    
    CreatePredictor --> LoadONNX[Load ONNX Model]
    LoadONNX --> InitSession[Initialize ONNX Session]
    InitSession --> LoadEncoder[Load Label Encoder]
    LoadEncoder --> GetClasses[Extract Class Names]
    GetClasses --> CreateTransform[Create Image Transform]
    CreateTransform --> Ready[Predictor Ready]
    
    Ready --> RegisterRoutes[Register FastAPI Routes]
    RegisterRoutes --> MountStatic[Mount Static Files]
    MountStatic --> MountTemplates[Setup Jinja2 Templates]
    MountTemplates --> StartServer[Start Uvicorn Server]
    StartServer --> Running([Server Running on Port 8000])
    
    Error1 --> End([Startup Failed])
    Error2 --> End
    
    style Start fill:#90EE90
    style Running fill:#90EE90
    style End fill:#FFB6C1
    style LoadONNX fill:#87CEEB
    style InitSession fill:#FFD700
```

### Image Validation Workflow

```mermaid
flowchart TD
    Start([Receive Uploaded File]) --> CheckExtension{Check File Extension}
    
    CheckExtension -->|Not in ALLOWED_EXTENSIONS| Error1[HTTPException: Invalid Format]
    CheckExtension -->|Valid| ReadFile[Read File Content]
    
    ReadFile --> CheckSize{File Size < 10MB?}
    CheckSize -->|No| Error2[HTTPException: File Too Large]
    CheckSize -->|Yes| TryOpen[Try Open with PIL]
    
    TryOpen --> CanOpen{Image Opens?}
    CanOpen -->|No| Error3[HTTPException: Invalid Image]
    CanOpen -->|Yes| VerifyImage[Verify Image Format]
    
    VerifyImage --> ValidImage{Valid Image?}
    ValidImage -->|No| Error3
    ValidImage -->|Yes| ResetPointer[Reset File Pointer]
    ResetPointer --> ReturnBytes[Return Image Bytes]
    
    Error1 --> End([Return Error])
    Error2 --> End
    Error3 --> End
    ReturnBytes --> End([Return Valid Bytes])
    
    style Start fill:#90EE90
    style ReturnBytes fill:#90EE90
    style End fill:#FFB6C1
    style Error1 fill:#FF6B6B
    style Error2 fill:#FF6B6B
    style Error3 fill:#FF6B6B
```

---

## Class-Level Details

### FoodClassificationApp (main.py)

**Purpose**: FastAPI application that serves the food classification web interface and REST API.

**Key Responsibilities**:
- Initialize FastAPI app with lifespan events
- Register HTTP routes (web UI and REST API)
- Handle image uploads and validation
- Coordinate between predictor and templates
- Serve static files and templates

**Key Attributes**:
- `app`: FastAPI application instance
- `predictor`: Global predictor instance (loaded at startup)
- `templates`: Jinja2 template engine
- `MAX_UPLOAD_SIZE`: Maximum file size (10MB)
- `ALLOWED_EXTENSIONS`: Allowed image formats

**Key Methods**:
- `lifespan()`: Startup/shutdown handler (loads model)
- `validate_image()`: Validates uploaded image file
- `home()`: Renders main page (GET /)
- `predict_web()`: Handles web form submission (POST /predict)
- `predict_api()`: REST API endpoint (POST /api/predict)
- `get_classes()`: Returns list of food classes (GET /api/classes)
- `health_check()`: Health check endpoint (GET /health)

**Endpoints**:
1. `GET /` - Web interface home page
2. `POST /predict` - Web form submission (returns HTML)
3. `POST /api/predict` - REST API (returns JSON)
4. `GET /api/classes` - List all food classes
5. `GET /health` - Health check

---

### FoodClassificationPredictor

**Purpose**: Service class that handles image preprocessing and ONNX model inference.

**Key Responsibilities**:
- Load ONNX model and label encoder at startup
- Preprocess images (resize, normalize, tensor conversion)
- Run ONNX inference
- Postprocess results (softmax, top-K selection, class mapping)
- Provide class name lookup

**Key Attributes**:
- `session`: ONNX Runtime InferenceSession
- `label_encoder`: sklearn LabelEncoder (loaded from joblib)
- `class_names`: List of food class names
- `transform`: torchvision transform pipeline
- `image_size`: Target image size (224, 224)
- `normalization_mean/std`: ImageNet normalization values

**Key Methods**:
- `load()`: Loads ONNX model and label encoder
- `preprocess_image()`: Converts PIL Image to ONNX-ready numpy array
- `predict()`: Main prediction method (returns dict with results)
- `predict_from_bytes()`: Convenience method for bytes input
- `get_class_names()`: Returns list of all class names

**Preprocessing Pipeline**:
1. Convert image to RGB (if needed)
2. Resize to (224, 224)
3. Convert to tensor (0-1 range)
4. Normalize with ImageNet mean/std
5. Add batch dimension [1, 3, 224, 224]
6. Convert to float32 numpy array

**Inference Pipeline**:
1. Get input name from ONNX session
2. Run session with preprocessed image
3. Get logits output
4. Apply softmax (numerically stable)
5. Get top-K predictions
6. Map indices to class names
7. Build response dictionary

---

### FoodPrediction (Pydantic Model)

**Purpose**: Response model for API validation and serialization.

**Key Fields**:
- `predicted_class`: Top predicted food class name
- `confidence`: Confidence score (0-1) for top prediction
- `top_predictions`: List of top K predictions (FoodPredictionItem)
- `all_probabilities`: Dictionary of all class probabilities
- `num_classes`: Total number of food classes

**Validation**: Automatically validates response structure and types.

---

### FoodPredictionItem (Pydantic Model)

**Purpose**: Single prediction item in top predictions list.

**Key Fields**:
- `class_name`: Food class name (aliased as `class`)
- `confidence`: Confidence score (0-1)
- `class_index`: Integer index of the class

---

## Data Flow Summary

### Request Flow

1. **Client** → HTTP Request (multipart/form-data with image)
2. **FastAPI** → Route handler receives request
3. **Validator** → Validates file (extension, size, format)
4. **Predictor** → Preprocesses image
5. **ONNX Runtime** → Runs inference on model
6. **Predictor** → Postprocesses results (softmax, top-K)
7. **Route** → Formats response (HTML or JSON)
8. **Client** ← HTTP Response with predictions

### Model Loading Flow

1. **Application Startup** → Lifespan handler triggered
2. **Predictor Creation** → Initialize with model paths
3. **ONNX Loading** → Load model.onnx into InferenceSession
4. **Encoder Loading** → Load label_encoder.joblib
5. **Class Extraction** → Get class names from encoder
6. **Transform Creation** → Create preprocessing pipeline
7. **Ready** → Predictor available for requests

---

## Key Design Patterns

1. **Singleton Pattern**: Global predictor instance (loaded once at startup)
2. **Service Layer Pattern**: Predictor encapsulates inference logic
3. **Template Method Pattern**: FastAPI defines request/response structure
4. **Factory Pattern**: Transform pipeline created from configuration
5. **Adapter Pattern**: Predictor adapts ONNX model to application needs

---

## Integration Points

### ONNX Model Integration

- **Model Format**: ONNX (Open Neural Network Exchange)
- **Runtime**: ONNX Runtime (ort.InferenceSession)
- **Input Shape**: [1, 3, 224, 224] (batch, channels, height, width)
- **Output Shape**: [num_classes] (logits)
- **Preprocessing**: Must match training preprocessing (resize, normalize)

### Label Encoder Integration

- **Format**: sklearn LabelEncoder saved with joblib
- **Purpose**: Maps class indices to class names
- **Loading**: Loaded at startup, used for all predictions
- **Dynamic**: Supports any number of classes (loaded from encoder)

### Web Interface Integration

- **Template Engine**: Jinja2
- **Static Files**: CSS, JavaScript (served via FastAPI static mount)
- **Image Display**: Base64-encoded images in HTML
- **Interactive**: Drag-and-drop upload, image preview, dynamic results

---

## Deployment Architecture

```mermaid
graph TB
    subgraph "Production Server"
        subgraph "Application Container"
            Uvicorn[Uvicorn Server<br/>Port 8000]
            FastAPI[FastAPI App]
            Predictor[Predictor Service]
        end
        
        subgraph "Model Storage"
            ONNXModel[ONNX Model File]
            LabelEncoder[Label Encoder File]
        end
        
        subgraph "Static Assets"
            Templates[HTML Templates]
            CSS[CSS Files]
            JS[JavaScript Files]
        end
    end
    
    subgraph "Clients"
        Browser[Web Browser]
        MobileApp[Mobile App]
        APIClient[API Client]
    end
    
    Browser --> Uvicorn
    MobileApp --> Uvicorn
    APIClient --> Uvicorn
    
    Uvicorn --> FastAPI
    FastAPI --> Predictor
    Predictor --> ONNXModel
    Predictor --> LabelEncoder
    FastAPI --> Templates
    FastAPI --> CSS
    FastAPI --> JS
    
    style Uvicorn fill:#667eea,color:#fff
    style FastAPI fill:#764ba2,color:#fff
    style Predictor fill:#f093fb,color:#fff
```

---

## Performance Considerations

1. **Model Loading**: Done once at startup (not per request)
2. **ONNX Runtime**: Optimized C++ backend for fast inference
3. **Image Preprocessing**: Efficient PIL and torchvision operations
4. **Batch Processing**: Single image per request (can be extended)
5. **Caching**: Predictor instance cached globally
6. **Async Support**: FastAPI async routes for concurrent requests

---

## Error Handling

1. **File Validation**: Extension, size, format checks
2. **Model Loading**: Startup validation with clear error messages
3. **Image Processing**: Graceful handling of corrupted images
4. **ONNX Inference**: Error handling for inference failures
5. **HTTP Errors**: Proper status codes (400, 500, 503)

---

*Last Updated: [Current Date]*
