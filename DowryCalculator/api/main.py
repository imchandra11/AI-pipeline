"""
FastAPI application for Dowry Calculator (Awareness-Only Project).
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import pandas as pd
from typing import Optional

from .predictor import DowryCalculatorPredictor
from .models import DowryInput, DowryPrediction

# Global predictor instance
predictor: Optional[DowryCalculatorPredictor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    global predictor
    try:
        # Setup paths - models are in DowryCalculator/models/
        DOWRY_DIR = Path(__file__).resolve().parent.parent
        MODELS_DIR = DOWRY_DIR / "models"
        
        predictor = DowryCalculatorPredictor(
            model_path=MODELS_DIR / "dowry_model.onnx",
            preprocessor_path=MODELS_DIR / "preprocessor.joblib"
        )
        predictor.load()
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown (cleanup if needed)
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Dowry Calculator API (Awareness-Only)",
    description="API for dowry amount prediction - For educational and awareness purposes only. Dowry is illegal in India.",
    version="1.0.0",
    lifespan=lifespan
)

# Setup paths
DOWRY_DIR = Path(__file__).resolve().parent.parent
API_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = API_DIR / "templates"
STATIC_DIR = API_DIR / "static"
DATA_DIR = DOWRY_DIR / "data"

# Mount static files and setup templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

def get_dropdown_options():
    """Get unique values for dropdown options from dataset."""
    dataset_path = DATA_DIR / "dowry_dataset.csv"
    if not dataset_path.exists():
        # Return default values if dataset not found
        return {
            "profession_options": ["Government Officer", "Software Engineer", "Doctor", "CA", "Business Owner", "Private Job", "Freelancer"],
            "education_options": ["PhD", "Masters", "Bachelors", "Diploma"],
            "location_options": ["Tier-1", "Tier-2", "Tier-3", "Rural"],
            "home_status_options": ["Own", "Rented"],
            "family_wealth_options": ["Lower", "Middle", "Upper-Middle", "Upper"],
            "marital_status_options": ["Single", "Married", "Divorced"],
            "marriage_type_options": ["Arranged", "Love"]
        }
    
    df = pd.read_csv(dataset_path)
    return {
        "profession_options": sorted(df["profession"].unique().tolist()),
        "education_options": sorted(df["education_level"].unique().tolist()),
        "location_options": sorted(df["location"].unique().tolist()),
        "home_status_options": sorted(df["home_status"].unique().tolist()),
        "family_wealth_options": sorted(df["family_wealth"].unique().tolist()),
        "marital_status_options": sorted(df["marital_status"].unique().tolist()),
        "marriage_type_options": sorted(df["marriage_type"].unique().tolist()),
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main prediction form with disclaimer."""
    options = get_dropdown_options()
    context = {
        "request": request,
        **options
    }
    return templates.TemplateResponse("index.html", context)

@app.post("/predict", response_class=HTMLResponse)
async def predict_web(
    request: Request,
    age: int = Form(...),
    monthly_salary: int = Form(...),
    profession: str = Form(...),
    education_level: str = Form(...),
    location: str = Form(...),
    home_status: str = Form(...),
    family_wealth: str = Form(...),
    marital_status: str = Form(...),
    marriage_type: str = Form(...),
    government_job: int = Form(...)
):
    """Handle web form prediction."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_data = {
            "age": age,
            "monthly_salary": monthly_salary,
            "profession": profession,
            "education_level": education_level,
            "location": location,
            "home_status": home_status,
            "family_wealth": family_wealth,
            "marital_status": marital_status,
            "marriage_type": marriage_type,
            "government_job": government_job
        }
        
        # Make prediction
        result = predictor.predict(input_data)
        
        # Get dropdown options for form
        options = get_dropdown_options()
        
        context = {
            "request": request,
            "prediction": result,
            "input_data": input_data,
            **options
        }
        
        return templates.TemplateResponse("index.html", context)
        
    except Exception as e:
        options = get_dropdown_options()
        context = {
            "request": request,
            "error": f"Prediction failed: {str(e)}",
            **options
        }
        return templates.TemplateResponse("index.html", context)

@app.post("/api/predict", response_model=DowryPrediction)
async def predict_api(input_data: DowryInput):
    """REST API endpoint for prediction."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = predictor.predict(input_data.model_dump())
        return DowryPrediction(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }
