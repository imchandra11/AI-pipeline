"""
FastAPI application for Food Classification.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Optional
from PIL import Image
import io

from .predictor import FoodClassificationPredictor
from .models import FoodPrediction

# Global predictor instance
predictor: Optional[FoodClassificationPredictor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    global predictor
    try:
        # Setup paths - models are in FoodClassification/models/
        FOOD_CLASSIFICATION_DIR = Path(__file__).resolve().parent.parent
        MODELS_DIR = FOOD_CLASSIFICATION_DIR / "models"
        
        predictor = FoodClassificationPredictor(
            model_path=MODELS_DIR / "food_classification_model.onnx",
            label_encoder_path=MODELS_DIR / "label_encoder.joblib",
            image_size=(224, 224),
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
        )
        predictor.load()
        print("✓ Models loaded successfully")
        print(f"✓ Loaded {len(predictor.get_class_names())} food classes: {', '.join(predictor.get_class_names())}")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown (cleanup if needed)
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Food Classification API",
    description="API for classifying food images using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Setup paths
FOOD_CLASSIFICATION_DIR = Path(__file__).resolve().parent.parent
API_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = API_DIR / "templates"
STATIC_DIR = API_DIR / "static"

# Mount static files and setup templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Configuration
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

def validate_image(file: UploadFile) -> bytes:
    """
    Validate and read image file.
    
    Args:
        file: Uploaded file
        
    Returns:
        Image bytes
        
    Raises:
        HTTPException if validation fails
    """
    # Check file extension
    file_ext = Path(file.filename or '').suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Read file content
    contents = file.file.read()
    
    # Check file size
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE / (1024*1024):.1f}MB"
        )
    
    # Validate image can be opened
    try:
        # Try to open and verify image
        image = Image.open(io.BytesIO(contents))
        image.verify()  # Verify it's a valid image (this closes the image)
        # Reopen image for actual use (verify() closes it)
        image = Image.open(io.BytesIO(contents))
        image.close()  # Close it, we'll reopen in predictor
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )
    
    # Reset file pointer
    file.file.seek(0)
    
    return contents

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main prediction form."""
    if predictor is None:
        context = {
            "request": request,
            "error": "Model not loaded. Please try again later.",
            "class_names": []
        }
    else:
        context = {
            "request": request,
            "class_names": predictor.get_class_names(),
            "num_classes": len(predictor.get_class_names())
        }
    return templates.TemplateResponse("index.html", context)

@app.post("/predict", response_class=HTMLResponse)
async def predict_web(
    request: Request,
    image: UploadFile = File(...),
):
    """Handle image upload and return prediction."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    try:
        # Validate and read image
        image_bytes = validate_image(image)
        
        # Load image
        image_obj = Image.open(io.BytesIO(image_bytes))
        
        # Get prediction
        num_classes = len(predictor.get_class_names())
        prediction = predictor.predict(image_obj, top_k=min(5, num_classes))
        
        # Prepare image for display (base64 encoded)
        import base64
        image_obj.seek(0) if hasattr(image_obj, 'seek') else None
        buffered = io.BytesIO()
        # Save in format that can be displayed
        image_obj = Image.open(io.BytesIO(image_bytes))  # Reopen to reset
        # Convert to RGB if needed
        if image_obj.mode != 'RGB':
            image_obj = image_obj.convert('RGB')
        image_obj.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        context = {
            "request": request,
            "prediction": prediction,
            "image_base64": img_base64,
            "image_filename": image.filename,
            "class_names": predictor.get_class_names(),
            "num_classes": len(predictor.get_class_names())
        }
        return templates.TemplateResponse("index.html", context)
    
    except HTTPException:
        raise
    except Exception as e:
        context = {
            "request": request,
            "error": str(e),
            "class_names": predictor.get_class_names() if predictor else [],
            "num_classes": len(predictor.get_class_names()) if predictor else 0
        }
        return templates.TemplateResponse("index.html", context)

@app.post("/api/predict", response_model=FoodPrediction)
async def predict_api(image: UploadFile = File(...)):
    """REST API endpoint for JSON predictions."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    try:
        # Validate and read image
        image_bytes = validate_image(image)
        
        # Load image
        image_obj = Image.open(io.BytesIO(image_bytes))
        
        # Get prediction
        num_classes = len(predictor.get_class_names())
        prediction = predictor.predict(image_obj, top_k=min(5, num_classes))
        
        return FoodPrediction(**prediction)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/classes")
async def get_classes():
    """Get list of all food classes."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    return {
        "classes": predictor.get_class_names(),
        "num_classes": len(predictor.get_class_names())
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "num_classes": len(predictor.get_class_names()) if predictor else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

