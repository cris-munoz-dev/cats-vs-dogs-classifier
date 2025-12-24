"""Quick test script to predict a single image."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.predict import Predictor
from src.config import CHECKPOINTS_DIR

def main():
    """Test prediction on a single image."""
    
    # Load best model
    model_path = CHECKPOINTS_DIR / "best_model.pth"
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first: python src/train.py")
        return
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    predictor = Predictor(model_path, device=str(device))
    
    print("=" * 60)
    print("Cats vs Dogs Predictor - Test Mode")
    print("=" * 60)
    print(f"\n✓ Model loaded from: {model_path}")
    print(f"✓ Using device: {device}")
    
    # Ask for image path
    print("\nEnter the path to an image to classify:")
    print("Example: data/raw/cats/cat.1.jpg")
    
    image_path = input("\nImage path: ").strip()
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"\n❌ Image not found: {image_path}")
        return
    
    # Make prediction
    print(f"\n🔍 Analyzing image: {image_path.name}")
    predicted_class, confidence = predictor.predict(image_path)
    
    # Display result
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"\n🎯 Predicted class: {predicted_class.upper()}")
    print(f"📊 Confidence: {confidence:.2%}")
    
    # Confidence interpretation
    if confidence > 0.95:
        print("💪 Very confident prediction!")
    elif confidence > 0.80:
        print("✅ Confident prediction")
    elif confidence > 0.60:
        print("⚠️  Moderate confidence")
    else:
        print("❓ Low confidence - image might be unclear")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
