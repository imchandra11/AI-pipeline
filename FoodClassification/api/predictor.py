"""
ONNX model predictor for food image classification.
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import Optional
import io

class FoodClassificationPredictor:
    """Predictor class for food classification using ONNX model."""
    
    def __init__(
        self,
        model_path: Path,
        label_encoder_path: Path,
        image_size: tuple[int, int] = (224, 224),
        normalization_mean: list[float] = [0.485, 0.456, 0.406],
        normalization_std: list[float] = [0.229, 0.224, 0.225],
    ):
        """
        Initialize the food classification predictor.
        
        Args:
            model_path: Path to ONNX model file
            label_encoder_path: Path to label encoder joblib file
            image_size: Target image size (height, width)
            normalization_mean: Normalization mean values
            normalization_std: Normalization std values
        """
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.image_size = image_size
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        
        self.session = None
        self.label_encoder = None
        self.class_names = None
        self.transform = None
    
    def load(self):
        """Load ONNX model, label encoder, and create transform."""
        import joblib
        
        # Load ONNX model
        self.session = ort.InferenceSession(str(self.model_path))
        
        # Load label encoder
        if self.label_encoder_path.exists():
            self.label_encoder = joblib.load(self.label_encoder_path)
            self.class_names = self.label_encoder.classes_.tolist()
        else:
            raise FileNotFoundError(f"Label encoder not found: {self.label_encoder_path}")
        
        # Create transform (same as validation/test - no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalization_mean, std=self.normalization_std),
        ])
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for inference.
        
        Args:
            image: PIL Image object (should already be opened)
            
        Returns:
            Preprocessed image as numpy array ready for ONNX
        """
        if self.transform is None:
            raise RuntimeError("Predictor not loaded. Call load() first.")
        
        # If image was verified (closed), reopen it
        # Image.verify() closes the image, so we need to reopen if needed
        if not hasattr(image, 'size') or image.size is None:
            # Image was closed by verify(), need to reopen from source
            raise ValueError("Image object is closed. Please provide an open Image object.")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Convert to numpy array and add batch dimension
        image_array = image_tensor.numpy()
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
        return image_array.astype(np.float32)
    
    def predict(self, image: Image.Image, top_k: Optional[int] = None) -> dict:
        """
        Make prediction on an image.
        
        Args:
            image: PIL Image object
            top_k: Number of top predictions to return (default: all classes or 5, whichever is smaller)
            
        Returns:
            Dictionary with prediction results including:
            - predicted_class: Top predicted class name
            - confidence: Confidence score for top prediction
            - top_predictions: List of top K predictions
            - all_probabilities: Dictionary with all class probabilities
        """
        if self.session is None or self.label_encoder is None:
            raise RuntimeError("Predictor not loaded. Call load() first.")
        
        # Set default top_k
        if top_k is None:
            top_k = min(5, len(self.class_names))
        else:
            top_k = min(top_k, len(self.class_names))
        
        # Preprocess image
        image_array = self.preprocess_image(image)
        
        # ONNX inference
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: image_array})
        logits = output[0][0]  # Get first (and only) sample
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Get top K predictions
        top_k_indices = np.argsort(probabilities)[::-1][:top_k]
        
        # Map indices to class names
        top_predictions = []
        for idx in top_k_indices:
            class_name = self.class_names[idx]
            probability = float(probabilities[idx])
            top_predictions.append({
                'class_name': class_name,
                'class': class_name,  # Alias for API compatibility
                'confidence': probability,
                'class_index': int(idx)
            })
        
        # Get all probabilities
        all_probabilities = {
            self.class_names[i]: float(prob)
            for i, prob in enumerate(probabilities)
        }
        
        # Top prediction
        top_prediction = top_predictions[0]
        
        return {
            "predicted_class": top_prediction['class'],
            "confidence": top_prediction['confidence'],
            "top_predictions": top_predictions,
            "all_probabilities": all_probabilities,
            "num_classes": len(self.class_names)
        }
    
    def predict_from_bytes(self, image_bytes: bytes, top_k: Optional[int] = None) -> dict:
        """
        Make prediction from image bytes.
        
        Args:
            image_bytes: Image file as bytes
            top_k: Number of top predictions to return (default: all classes or 5, whichever is smaller)
            
        Returns:
            Dictionary with prediction results
        """
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict(image, top_k=top_k)
    
    def get_class_names(self) -> list[str]:
        """Get list of all class names."""
        if self.class_names is None:
            raise RuntimeError("Predictor not loaded. Call load() first.")
        return self.class_names.copy()

