"""Inference script for making predictions on new images."""

import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import logging
from typing import Tuple

from src.models.model import create_model
from src.config import IMAGE_SIZE, MEAN, STD, CLASS_NAMES, CHECKPOINTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:
    """Predictor class for inference on single images."""
    
    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
        class_names: list = CLASS_NAMES
    ):
        """
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on
            class_names: List of class names
        """
        self.device = device
        self.class_names = class_names
        
        # Load model
        self.model = create_model(device=device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, image_path: Path) -> Tuple[str, float]:
        """Make prediction on a single image.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = probabilities.max(1)
        
        predicted_class = self.class_names[predicted_idx.item()]
        confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    
    def predict_batch(self, image_paths: list) -> list:
        """Make predictions on multiple images.
        
        Args:
            image_paths: List of paths to image files
        
        Returns:
            List of tuples (predicted_class, confidence)
        """
        results = []
        for image_path in image_paths:
            pred_class, confidence = self.predict(image_path)
            results.append((pred_class, confidence))
        
        return results


def main():
    """Example usage of the Predictor."""
    # Initialize predictor
    model_path = CHECKPOINTS_DIR / "best_model.pth"
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.info("Please train the model first using: python src/train.py")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = Predictor(model_path, device=device)
    
    # Example: predict on a single image
    # Uncomment and modify the path below to test
    # image_path = Path("path/to/your/image.jpg")
    # predicted_class, confidence = predictor.predict(image_path)
    # logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
