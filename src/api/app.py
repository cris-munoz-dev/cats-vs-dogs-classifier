"""FastAPI application for serving the Cats vs Dogs classifier."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from pathlib import Path
import logging

from src.models.model import create_model
from src.predict import Predictor
from src.config import CHECKPOINTS_DIR, CLASS_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="API for classifying images of cats and dogs using deep learning",
    version="1.0.0"
)

# Global predictor instance
predictor = None


@app.on_event("startup")
async def load_model():
    """Load the model on startup."""
    global predictor
    
    model_path = CHECKPOINTS_DIR / "best_model.pth"
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.warning("API will start but predictions will fail until model is trained")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = Predictor(model_path, device=device)
    logger.info("Model loaded successfully")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Cats vs Dogs Classifier API",
        "endpoints": {
            "/predict": "POST - Upload an image to classify",
            "/health": "GET - Check API health",
            "/info": "GET - Get model information"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = predictor is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }


@app.get("/info")
async def model_info():
    """Get model information."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model": "ResNet50 with Transfer Learning",
        "classes": CLASS_NAMES,
        "input_size": "224x224",
        "framework": "PyTorch"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded image.
    
    Args:
        file: Uploaded image file
    
    Returns:
        JSON response with prediction and confidence
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Save temporarily
        temp_path = Path("/tmp/temp_image.jpg")
        image.save(temp_path)
        
        # Make prediction
        predicted_class, confidence = predictor.predict(temp_path)
        
        # Clean up
        temp_path.unlink()
        
        return JSONResponse(content={
            "prediction": predicted_class,
            "confidence": float(confidence),
            "all_classes": {
                CLASS_NAMES[0]: float(1 - confidence) if predicted_class == CLASS_NAMES[1] else float(confidence),
                CLASS_NAMES[1]: float(confidence) if predicted_class == CLASS_NAMES[1] else float(1 - confidence)
            }
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
