"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional

class DowryInput(BaseModel):
    """Input model for dowry amount prediction."""
    age: int = Field(..., ge=18, le=100, description="Age of the person")
    monthly_salary: int = Field(..., ge=0, description="Monthly salary in INR")
    profession: str = Field(..., description="Profession type")
    education_level: str = Field(..., description="Education level")
    location: str = Field(..., description="Location tier")
    home_status: str = Field(..., description="Home ownership status")
    family_wealth: str = Field(..., description="Family wealth category")
    marital_status: str = Field(..., description="Marital status")
    marriage_type: str = Field(..., description="Marriage type")
    government_job: int = Field(..., ge=0, le=1, description="Government job indicator (0 or 1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 28,
                "monthly_salary": 80000,
                "profession": "Software Engineer",
                "education_level": "Masters",
                "location": "Tier-1",
                "home_status": "Own",
                "family_wealth": "Upper-Middle",
                "marital_status": "Single",
                "marriage_type": "Arranged",
                "government_job": 0
            }
        }

class DowryPrediction(BaseModel):
    """Output model for dowry amount prediction."""
    dowry_amount_lakhs: float = Field(..., description="Predicted dowry amount in lakhs")
    dowry_amount_rupees: float = Field(..., description="Predicted dowry amount in rupees")
    formatted_amount: str = Field(..., description="Formatted amount string")
