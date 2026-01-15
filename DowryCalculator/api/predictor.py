"""
ONNX model predictor for dowry amount prediction (regression).
"""

import onnxruntime as ort
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any

class DowryCalculatorPredictor:
    """Predictor class for dowry amount prediction using ONNX model."""
    
    def __init__(
        self,
        model_path: Path,
        preprocessor_path: Path
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.session = None
        self.preprocessor = None
        
    def load(self):
        """Load ONNX model and preprocessor."""
        # Load ONNX model
        self.session = ort.InferenceSession(str(self.model_path))
        
        # Load preprocessor
        self.preprocessor = joblib.load(self.preprocessor_path)
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction on input data.
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Dictionary with prediction results
        """
        if self.session is None or self.preprocessor is None:
            raise RuntimeError("Predictor not loaded. Call load() first.")
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Get feature columns (categorical + numeric) - must match training config
        categorical_cols = [
            'profession',
            'education_level',
            'location',
            'home_status',
            'family_wealth',
            'marital_status',
            'marriage_type'
        ]
        numeric_cols = [
            'age',
            'monthly_salary',
            'government_job'
        ]
        feature_cols = categorical_cols + numeric_cols
        
        # Preprocess
        features = df[feature_cols]
        transformed = self.preprocessor.transform(features)
        
        # ONNX inference
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: transformed.astype(np.float32)})
        
        # Regression: single value output
        predicted_amount = float(output[0][0][0])
        
        # Ensure non-negative
        predicted_amount = max(0.0, predicted_amount)
        
        return {
            "dowry_amount_lakhs": round(predicted_amount, 2),
            "dowry_amount_rupees": round(predicted_amount * 100000, 2),
            "formatted_amount": f"₹{predicted_amount:.2f} Lakhs"
        }
