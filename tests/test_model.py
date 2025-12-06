"""Unit tests for the model module."""

import pytest
import torch
from src.models.model import CatsDogsClassifier, create_model, count_parameters


def test_model_creation():
    """Test model creation."""
    model = CatsDogsClassifier(num_classes=2, pretrained=False)
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_model_forward():
    """Test model forward pass."""
    model = CatsDogsClassifier(num_classes=2, pretrained=False)
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    output = model(x)
    
    assert output.shape == (batch_size, 2)


def test_freeze_backbone():
    """Test backbone freezing."""
    model = CatsDogsClassifier(num_classes=2, pretrained=False, freeze_backbone=True)
    
    # Check that most parameters are frozen
    frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
    total_params = sum(1 for p in model.parameters())
    
    assert frozen_params > 0
    assert frozen_params < total_params


def test_unfreeze_backbone():
    """Test backbone unfreezing."""
    model = CatsDogsClassifier(num_classes=2, pretrained=False, freeze_backbone=True)
    model.unfreeze_backbone()
    
    # Check that all parameters are trainable
    trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
    total_params = sum(1 for p in model.parameters())
    
    assert trainable_params == total_params


def test_count_parameters():
    """Test parameter counting."""
    model = CatsDogsClassifier(num_classes=2, pretrained=False, freeze_backbone=True)
    params = count_parameters(model)
    
    assert 'trainable' in params
    assert 'total' in params
    assert 'frozen' in params
    assert params['total'] == params['trainable'] + params['frozen']


def test_create_model_factory():
    """Test model factory function."""
    model = create_model(num_classes=2, pretrained=False, device='cpu')
    assert model is not None
    assert next(model.parameters()).device.type == 'cpu'
