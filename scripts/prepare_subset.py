"""Script to create a subset of the dataset for quick testing."""

import shutil
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RAW_DATA_DIR


def create_subset(
    source_cats_dir: Path,
    source_dogs_dir: Path,
    num_samples: int = 1000
):
    """Create a subset of the dataset.
    
    Args:
        source_cats_dir: Directory with all cat images
        source_dogs_dir: Directory with all dog images
        num_samples: Number of samples per class to copy
    """
    # Create destination directories
    dest_cats_dir = RAW_DATA_DIR / "cats"
    dest_dogs_dir = RAW_DATA_DIR / "dogs"
    
    dest_cats_dir.mkdir(parents=True, exist_ok=True)
    dest_dogs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating subset with {num_samples} images per class...")
    print(f"Destination: {RAW_DATA_DIR}")
    
    # Copy cat images
    cat_images = sorted(list(source_cats_dir.glob("*.jpg")))[:num_samples]
    print(f"\nCopying {len(cat_images)} cat images...")
    for i, img_path in enumerate(cat_images, 1):
        dest_path = dest_cats_dir / img_path.name
        if not dest_path.exists():
            shutil.copy2(img_path, dest_path)
        if i % 100 == 0:
            print(f"  Copied {i}/{num_samples} cats")
    
    # Copy dog images
    dog_images = sorted(list(source_dogs_dir.glob("*.jpg")))[:num_samples]
    print(f"\nCopying {len(dog_images)} dog images...")
    for i, img_path in enumerate(dog_images, 1):
        dest_path = dest_dogs_dir / img_path.name
        if not dest_path.exists():
            shutil.copy2(img_path, dest_path)
        if i % 100 == 0:
            print(f"  Copied {i}/{num_samples} dogs")
    
    print(f"\n✅ Subset created successfully!")
    print(f"   Cats: {len(list(dest_cats_dir.glob('*.jpg')))} images")
    print(f"   Dogs: {len(list(dest_dogs_dir.glob('*.jpg')))} images")
    print(f"   Total: {len(list(dest_cats_dir.glob('*.jpg'))) + len(list(dest_dogs_dir.glob('*.jpg')))} images")


if __name__ == "__main__":
    print("=" * 60)
    print("Dataset Subset Creator")
    print("=" * 60)
    
    # Ask user for source directories
    print("\nWhere are your downloaded images?")
    print("Example: /Users/rmunozp/Downloads/cats-and-dogs")
    
    source_base = input("\nEnter the base directory path: ").strip()
    source_base = Path(source_base)
    
    if not source_base.exists():
        print(f"❌ Directory not found: {source_base}")
        sys.exit(1)
    
    # Try to find cats and dogs directories
    source_cats = source_base / "cats"
    source_dogs = source_base / "dogs"
    
    # Alternative: PetImages structure (Microsoft dataset)
    if not source_cats.exists():
        source_cats = source_base / "Cat"
    if not source_dogs.exists():
        source_dogs = source_base / "Dog"
    
    # Alternative: train structure (Kaggle dataset)
    if not source_cats.exists():
        source_cats = source_base / "train" / "cats"
    if not source_dogs.exists():
        source_dogs = source_base / "train" / "dogs"
    
    # Check if directories exist
    if not source_cats.exists() or not source_dogs.exists():
        print(f"\n❌ Could not find cats/dogs directories in {source_base}")
        print("\nPlease organize your images as:")
        print(f"  {source_base}/cats/")
        print(f"  {source_base}/dogs/")
        sys.exit(1)
    
    # Count images
    num_cats = len(list(source_cats.glob("*.jpg")))
    num_dogs = len(list(source_dogs.glob("*.jpg")))
    
    print(f"\n✓ Found {num_cats} cat images in: {source_cats}")
    print(f"✓ Found {num_dogs} dog images in: {source_dogs}")
    
    # Ask for number of samples
    print(f"\nHow many images per class do you want to use?")
    print(f"Recommended: 1000 (for quick testing)")
    print(f"Maximum: {min(num_cats, num_dogs)}")
    
    num_samples = input(f"\nEnter number (default: 1000): ").strip()
    num_samples = int(num_samples) if num_samples else 1000
    
    if num_samples > min(num_cats, num_dogs):
        print(f"⚠️  Requested {num_samples} but only {min(num_cats, num_dogs)} available")
        num_samples = min(num_cats, num_dogs)
    
    # Create subset
    create_subset(source_cats, source_dogs, num_samples)
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Verify setup: python scripts/quick_start.py")
    print("2. Train model: python src/train.py")
    print("3. Monitor training: tensorboard --logdir=logs")
    print("=" * 60)
