"""Quick start script to verify installation and setup."""

import sys
import torch
import torchvision
from pathlib import Path

print("=" * 60)
print("Cats vs Dogs Classifier - Quick Start Check")
print("=" * 60)

# Check Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")

# Check PyTorch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ Torchvision version: {torchvision.__version__}")

# Check CUDA availability
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("✓ CUDA not available (will use CPU)")

# Check MPS (Apple Silicon)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("✓ Apple MPS available")

# Check project structure
from src.config import PROJECT_ROOT, DATA_DIR, MODELS_DIR, LOGS_DIR

print(f"\n✓ Project root: {PROJECT_ROOT}")
print(f"✓ Data directory: {DATA_DIR}")
print(f"✓ Models directory: {MODELS_DIR}")
print(f"✓ Logs directory: {LOGS_DIR}")

# Check if data exists
cats_dir = DATA_DIR / "raw" / "cats"
dogs_dir = DATA_DIR / "raw" / "dogs"

if cats_dir.exists() and dogs_dir.exists():
    num_cats = len(list(cats_dir.glob("*.jpg")))
    num_dogs = len(list(dogs_dir.glob("*.jpg")))
    print(f"\n✓ Dataset found:")
    print(f"  - Cats: {num_cats} images")
    print(f"  - Dogs: {num_dogs} images")
else:
    print("\n⚠ Dataset not found!")
    print("  Run: python scripts/download_data.py for instructions")

# Test model creation
print("\n✓ Testing model creation...")
from src.models.model import create_model
model = create_model(pretrained=False, device='cpu')
print("  Model created successfully!")

# Test data transforms
print("✓ Testing data transforms...")
from src.data.dataset import get_transforms
transform = get_transforms(train=True)
print("  Transforms created successfully!")

print("\n" + "=" * 60)
print("Setup verification complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Download the dataset (see scripts/download_data.py)")
print("2. Train the model: python src/train.py")
print("3. Evaluate: python src/evaluate.py")
print("4. Deploy API: python src/api/app.py")
print("=" * 60)
