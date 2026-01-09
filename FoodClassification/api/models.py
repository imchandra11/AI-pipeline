"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, List

class FoodPredictionItem(BaseModel):
    """Single prediction item."""
    class_name: str = Field(..., description="Class name", alias="class")
    confidence: float = Field(..., description="Confidence score (0-1)")
    class_index: int = Field(..., description="Class index")
    
    class Config:
        populate_by_name = True

class FoodPrediction(BaseModel):
    """Output model for food classification prediction."""
    predicted_class: str = Field(..., description="Top predicted food class")
    confidence: float = Field(..., description="Confidence score for top prediction (0-1)")
    top_predictions: List[FoodPredictionItem] = Field(..., description="Top K predictions")
    all_probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")
    num_classes: int = Field(..., description="Total number of food classes")

