"""Configuration settings for the Cats vs Dogs classifier."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model configuration
MODEL_NAME = "resnet50"
NUM_CLASSES = 2  # cats and dogs
PRETRAINED = True

# Training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = 4

# Image configuration
IMAGE_SIZE = 224  # ResNet input size
MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
STD = [0.229, 0.224, 0.225]   # ImageNet std

# Data split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Device configuration
# Priority: CUDA (NVIDIA) > MPS (Apple) > CPU
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Logging
LOG_INTERVAL = 10  # Log every N batches
SAVE_INTERVAL = 1  # Save checkpoint every N epochs

# Class names
CLASS_NAMES = ["cat", "dog"]
