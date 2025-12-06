"""Unit tests for the data module."""

import pytest
import torch
from pathlib import Path
from src.data.dataset import CatsDogsDataset, get_transforms


def test_get_transforms_train():
    """Test training transforms."""
    transform = get_transforms(train=True)
    assert transform is not None


def test_get_transforms_val():
    """Test validation transforms."""
    transform = get_transforms(train=False)
    assert transform is not None


def test_dataset_creation():
    """Test dataset creation (will fail if no data)."""
    # This test will only pass if you have data
    # You can skip it or mock the data
    pass


def test_transform_output_shape():
    """Test that transforms produce correct output shape."""
    from PIL import Image
    import numpy as np
    
    # Create a dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    
    transform = get_transforms(train=False)
    transformed = transform(dummy_image)
    
    assert transformed.shape == (3, 224, 224)
    assert isinstance(transformed, torch.Tensor)
