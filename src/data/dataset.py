"""Dataset and DataLoader utilities for Cats vs Dogs classification."""

import os
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import logging

from src.config import (
    IMAGE_SIZE, MEAN, STD, BATCH_SIZE, NUM_WORKERS,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CatsDogsDataset(Dataset):
    """Custom Dataset for Cats vs Dogs classification.
    
    Expected directory structure:
        data/raw/
            cats/
                cat.1.jpg
                cat.2.jpg
                ...
            dogs/
                dog.1.jpg
                dog.2.jpg
                ...
    """
    
    def __init__(self, root_dir: Path, transform=None):
        """
        Args:
            root_dir: Path to the data directory containing 'cats' and 'dogs' folders
            transform: Optional transform to be applied on images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {"cats": 0, "dogs": 1}
        
        # Load all image paths and labels
        self._load_samples()
        
    def _load_samples(self):
        """Load all image paths and their corresponding labels."""
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Directory not found: {class_dir}")
                continue
                
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((img_path, class_idx))
        
        logger.info(f"Loaded {len(self.samples)} images")
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(train: bool = True) -> transforms.Compose:
    """Get image transformations for training or validation.
    
    Args:
        train: If True, returns training transforms with augmentation.
               If False, returns validation transforms without augmentation.
    
    Returns:
        Composed transforms
    """
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])


def create_dataloaders(
    data_dir: Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Path to the data directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    full_dataset = CatsDogsDataset(
        root_dir=data_dir,
        transform=get_transforms(train=False)  # Will be replaced for train set
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(TRAIN_SPLIT * total_size)
    val_size = int(VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply training transforms to train dataset
    train_dataset.dataset.transform = get_transforms(train=True)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    return train_loader, val_loader, test_loader
