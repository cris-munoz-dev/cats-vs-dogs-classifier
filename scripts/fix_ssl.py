"""Fix SSL certificate issues for downloading pre-trained models."""

import ssl
import certifi

# This script helps fix SSL certificate verification errors
# when downloading pre-trained models from PyTorch

print("SSL Certificate Fix")
print("=" * 60)
print("\nThis script will help you fix SSL certificate errors.")
print("\nOption 1: Install certifi certificates (Recommended)")
print("Run this command:")
print("  /Applications/Python\\ 3.10/Install\\ Certificates.command")
print("\nOption 2: Temporarily disable SSL verification (Not recommended)")
print("Add this to the top of train.py:")
print("  import ssl")
print("  ssl._create_default_https_context = ssl._create_unverified_context")
print("\nOption 3: Download model manually")
print("Download from: https://download.pytorch.org/models/resnet50-0676ba61.pth")
print("Save to: ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth")
print("=" * 60)
