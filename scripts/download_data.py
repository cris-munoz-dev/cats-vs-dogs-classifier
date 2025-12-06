"""Script to download and prepare the Cats vs Dogs dataset."""

import os
import zipfile
import requests
from pathlib import Path
import shutil
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, destination: Path):
    """Download a file with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)


def prepare_dataset(data_dir: Path, num_samples_per_class: int = 1000):
    """Prepare the Cats vs Dogs dataset.
    
    This is a helper script. You'll need to:
    1. Download the dataset from Kaggle or Microsoft
    2. Extract it
    3. Organize into cats/ and dogs/ folders
    
    Args:
        data_dir: Directory to save the dataset
        num_samples_per_class: Number of samples per class (for testing)
    """
    logger.info("Dataset preparation instructions:")
    logger.info("\n1. Download the dataset from one of these sources:")
    logger.info("   - Kaggle: https://www.kaggle.com/c/dogs-vs-cats/data")
    logger.info("   - Microsoft: https://www.microsoft.com/en-us/download/details.aspx?id=54765")
    logger.info("\n2. Extract the downloaded file")
    logger.info("\n3. Organize images into this structure:")
    logger.info(f"   {data_dir}/")
    logger.info("   ├── cats/")
    logger.info("   │   ├── cat.1.jpg")
    logger.info("   │   ├── cat.2.jpg")
    logger.info("   │   └── ...")
    logger.info("   └── dogs/")
    logger.info("       ├── dog.1.jpg")
    logger.info("       ├── dog.2.jpg")
    logger.info("       └── ...")
    logger.info("\nNote: You'll need a Kaggle account to download the dataset.")


if __name__ == "__main__":
    from src.config import RAW_DATA_DIR
    prepare_dataset(RAW_DATA_DIR)
