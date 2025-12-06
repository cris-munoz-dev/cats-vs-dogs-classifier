"""Model architecture using Transfer Learning with ResNet50."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
import logging

from src.config import MODEL_NAME, NUM_CLASSES, PRETRAINED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CatsDogsClassifier(nn.Module):
    """Cats vs Dogs classifier using transfer learning with ResNet50.
    
    This model uses a pre-trained ResNet50 as the backbone and replaces
    the final fully connected layer for binary classification.
    """
    
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = PRETRAINED,
        freeze_backbone: bool = True
    ):
        """
        Args:
            num_classes: Number of output classes (2 for cats vs dogs)
            pretrained: Whether to use pre-trained ImageNet weights
            freeze_backbone: If True, freeze all layers except the final FC layer
        """
        super(CatsDogsClassifier, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone layers frozen")
        
        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
        logger.info(f"Model initialized with {num_classes} classes")
        logger.info(f"Pretrained: {pretrained}, Freeze backbone: {freeze_backbone}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """Unfreeze the backbone layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the end.
                       If None, unfreezes all layers.
        """
        if num_layers is None:
            # Unfreeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("All backbone layers unfrozen")
        else:
            # Unfreeze last n layers
            layers = list(self.backbone.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info(f"Last {num_layers} backbone layers unfrozen")


def create_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = PRETRAINED,
    freeze_backbone: bool = True,
    device: str = "cpu"
) -> CatsDogsClassifier:
    """Factory function to create and initialize the model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pre-trained weights
        freeze_backbone: Whether to freeze backbone layers
        device: Device to move the model to ('cpu' or 'cuda')
    
    Returns:
        Initialized model on the specified device
    """
    model = CatsDogsClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    model = model.to(device)
    logger.info(f"Model moved to {device}")
    
    return model


def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters in the model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "trainable": trainable_params,
        "total": total_params,
        "frozen": total_params - trainable_params
    }
