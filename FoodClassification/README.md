# Food Classification API & Web Interface

A FastAPI-based web application for food image classification using deep learning. Upload food images and get instant predictions with confidence scores for all food classes.

## Features

- **Web Interface**: User-friendly drag-and-drop image upload with real-time preview
- **REST API**: JSON API for programmatic access
- **Dynamic Class Loading**: Automatically supports any number of food classes (scales from 3 to 100+ classes)
- **Visual Predictions**: Beautiful probability visualizations for all classes
- **Production Ready**: ONNX model inference for fast, optimized predictions
- **Modern UI**: Responsive design that works on desktop and mobile

## Project Structure

```
FoodClassification/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic validation models
│   ├── predictor.py         # ONNX inference logic
│   ├── templates/
│   │   ├── base.html        # Base template
│   │   └── index.html       # Main upload interface
│   └── static/
│       └── css/
│           └── style.css    # Styling
├── models/
│   ├── food_classification_model.onnx    # Trained ONNX model
│   └── label_encoder.joblib              # Class label encoder
├── data/
│   └── images/              # Training dataset (class folders)
├── configs/
│   ├── food.yaml            # Training configuration
│   └── food.local.yaml      # Local overrides
├── lightning_logs/          # Training logs
└── README.md                # This file
```

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- Trained model files:
  - `models/food_classification_model.onnx`
  - `models/label_encoder.joblib`

## Installation

### 1. Install Dependencies

From the project root directory:

```bash
pip install -r requirements.txt
```

Required packages (should already be in requirements.txt):
- `fastapi>=0.104.0`
- `uvicorn[standard]>=0.24.0`
- `jinja2>=3.1.0`
- `python-multipart>=0.0.6`
- `onnxruntime>=1.16.0`
- `Pillow>=9.0.0`
- `torchvision>=0.15.0`

### 2. Verify Model Files

Ensure these files exist:
- `FoodClassification/models/food_classification_model.onnx`
- `FoodClassification/models/label_encoder.joblib`

If you haven't trained a model yet, train it first using:

```bash
PYTHONPATH=. python ImageClassification/mainfittest.py --config FoodClassification/configs/food.yaml
```

## Running the API

### Option 1: Direct Python (from project root)

```bash
python -m FoodClassification.api.main
```

### Option 2: Uvicorn (Recommended)

```bash
# Basic
uvicorn FoodClassification.api.main:app --host 0.0.0.0 --port 8000

# With auto-reload (for development)
uvicorn FoodClassification.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: Windows PowerShell

```powershell
cd C:\Users\91838\Desktop\AI-pipeline
python -m FoodClassification.api.main
```

## Access Points

Once the server is running, access:

- **Web Interface**: http://localhost:8000/
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Classes List**: http://localhost:8000/api/classes

## Usage

### Web Interface

1. Open http://localhost:8000/ in your browser
2. **Drag and drop** a food image onto the upload area, or **click to browse**
3. See image preview before submitting
4. Click **"Classify Food"** to get predictions
5. View results:
   - Top prediction with confidence
   - Top 3-5 predictions with probability bars
   - All class probabilities (scrollable for many classes)

### REST API

#### 1. Upload Image and Get Prediction

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@path/to/your/food_image.jpg"
```

**Response:**
```json
{
  "predicted_class": "samosa",
  "confidence": 0.95,
  "top_predictions": [
    {
      "class_name": "samosa",
      "confidence": 0.95,
      "class_index": 2
    },
    {
      "class_name": "jalebi",
      "confidence": 0.03,
      "class_index": 1
    },
    {
      "class_name": "chai",
      "confidence": 0.02,
      "class_index": 0
    }
  ],
  "all_probabilities": {
    "chai": 0.02,
    "jalebi": 0.03,
    "samosa": 0.95
  },
  "num_classes": 3
}
```

#### 2. Get All Food Classes

```bash
curl "http://localhost:8000/api/classes"
```

**Response:**
```json
{
  "classes": ["chai", "jalebi", "samosa"],
  "num_classes": 3
}
```

#### 3. Health Check

```bash
curl "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_classes": 3
}
```

### Python Client Example

```python
import requests

# Upload image
with open("food_image.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post(
        "http://localhost:8000/api/predict",
        files=files
    )

result = response.json()
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence'] * 100:.1f}%")
print("\nTop Predictions:")
for pred in result['top_predictions']:
    print(f"  {pred['class_name']}: {pred['confidence'] * 100:.1f}%")
```

### JavaScript/TypeScript Example

```javascript
// Upload image using fetch
const formData = new FormData();
formData.append('image', fileInput.files[0]);

const response = await fetch('http://localhost:8000/api/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log('Predicted:', result.predicted_class);
console.log('Confidence:', result.confidence);
```

## API Endpoints

### `GET /`
Returns the web interface HTML page.

### `POST /predict`
Web form endpoint for image upload (returns HTML page with results).

**Request:**
- Content-Type: `multipart/form-data`
- Form field: `image` (file)

**Response:** HTML page with prediction results

### `POST /api/predict`
REST API endpoint for image classification.

**Request:**
- Content-Type: `multipart/form-data`
- Form field: `image` (file)

**Response:** JSON
```json
{
  "predicted_class": "string",
  "confidence": 0.0-1.0,
  "top_predictions": [...],
  "all_probabilities": {...},
  "num_classes": int
}
```

### `GET /api/classes`
Get list of all food classes.

**Response:**
```json
{
  "classes": ["class1", "class2", ...],
  "num_classes": int
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true/false,
  "num_classes": int
}
```

### `GET /docs`
Interactive API documentation (Swagger UI).

### `GET /redoc`
Alternative API documentation (ReDoc).

## Supported Image Formats

- JPEG/JPG
- PNG
- WEBP
- BMP

**Maximum file size:** 10MB

## Configuration

The API automatically loads configuration from:
- Image size: 224x224 pixels (matches training)
- Normalization: ImageNet defaults
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

To change these, modify `FoodClassification/api/main.py` in the `lifespan` function.

## Adding More Food Classes

The API automatically scales to any number of classes:

1. **Retrain your model** with the new dataset including additional food classes
2. **Ensure the label encoder is saved** during training (should be automatic)
3. **Restart the API** - it will automatically load the new classes from `label_encoder.joblib`

No code changes needed! The web interface and API will automatically adapt to the new number of classes.

## Troubleshooting

### Error: "Model not loaded"

**Solution:**
- Check that `models/food_classification_model.onnx` exists
- Check that `models/label_encoder.joblib` exists
- Verify file paths in the startup logs
- Ensure you're running from the project root directory

### Error: "Invalid image file"

**Possible causes:**
- Unsupported image format
- Corrupted image file
- File too large (>10MB)

**Solution:**
- Convert image to JPEG or PNG
- Reduce image file size
- Try a different image

### Error: Import errors

**Solution:**
```bash
# Ensure you're in project root
cd C:\Users\91838\Desktop\AI-pipeline

# Verify PYTHONPATH if needed (usually not required)
# Windows PowerShell:
$env:PYTHONPATH = "."

# Install/update dependencies
pip install -r requirements.txt
```

### Slow predictions

**Optimizations:**
- Use GPU for ONNX inference (install `onnxruntime-gpu` instead of `onnxruntime`)
- Reduce image size before upload
- Use smaller batch sizes if processing multiple images

### Port already in use

**Solution:**
Change the port:
```bash
uvicorn FoodClassification.api.main:app --port 8001
```

Or find and stop the process using port 8000.

## Development

### Making Changes

1. **API Code**: Edit files in `FoodClassification/api/`
2. **Templates**: Edit `FoodClassification/api/templates/`
3. **Styling**: Edit `FoodClassification/api/static/css/style.css`

### Testing Changes

If running with `--reload` flag, changes will auto-reload. Otherwise, restart the server.

### Debugging

Enable debug mode:
```bash
uvicorn FoodClassification.api.main:app --reload --log-level debug
```

Check server logs for detailed error messages.

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn FoodClassification.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker (Example)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "FoodClassification.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

You can configure via environment variables:
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)
- `MAX_UPLOAD_SIZE`: Max upload size in bytes (default: 10485760 = 10MB)

## Performance Tips

1. **Use GPU**: Install `onnxruntime-gpu` for faster inference
2. **Image Preprocessing**: Images are automatically resized to 224x224
3. **Caching**: Model is loaded once at startup and reused
4. **Batch Processing**: For multiple images, send separate requests or implement batch endpoint

## Security Considerations

- **File Upload Validation**: File type and size are validated
- **Image Verification**: Images are verified to be valid before processing
- **CORS**: Configure CORS if accessing from different domains
- **Rate Limiting**: Consider adding rate limiting for production use

## Examples

### Example 1: Classify a Food Image

**Using curl:**
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "image=@samosa.jpg"
```

**Using Python:**
```python
import requests

url = "http://localhost:8000/api/predict"
files = {"image": open("samosa.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Example 2: Get All Classes

```python
import requests

response = requests.get("http://localhost:8000/api/classes")
classes = response.json()["classes"]
print(f"Available food classes: {', '.join(classes)}")
```

## Model Information

- **Model Type**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224 RGB images
- **Output**: Logits for each food class (softmax applied for probabilities)
- **Framework**: PyTorch → ONNX

## Training the Model

To train or retrain the model:

```bash
# From project root
PYTHONPATH=. python ImageClassification/mainfittest.py \
  --config FoodClassification/configs/food.yaml \
  --config FoodClassification/configs/food.local.yaml
```

See `README_IMAGE_CLASSIFICATION.md` in the project root for detailed training instructions.

## License

Same as main project.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review server logs for error messages
3. Verify model files exist and are valid
4. Check API documentation at `/docs` endpoint

## Changelog

### Version 1.0.0
- Initial release
- Web interface with drag-drop upload
- REST API endpoints
- Dynamic class loading
- Probability visualization
- Production-ready ONNX inference

