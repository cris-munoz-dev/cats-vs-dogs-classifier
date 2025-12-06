"""
Jupyter notebook for data exploration and visualization.

This notebook helps you:
1. Explore the dataset
2. Visualize sample images
3. Check class distribution
4. Test data augmentation
"""

# Import libraries
import sys
sys.path.append('..')

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from src.data.dataset import CatsDogsDataset, get_transforms
from src.config import RAW_DATA_DIR, CLASS_NAMES

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')

print("Notebook ready! Start exploring your data.")
