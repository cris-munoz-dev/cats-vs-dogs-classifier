# Cats vs Dogs Image Classifier 🐱🐶

A professional deep learning project for classifying images of cats and dogs using **PyTorch** and **Transfer Learning** with ResNet50.

## 🎯 Project Overview

This project demonstrates:
- **Transfer Learning** with pre-trained ResNet50
- **Data augmentation** for better generalization
- **Model training** with TensorBoard logging
- **Model evaluation** with metrics and visualizations
- **REST API deployment** with FastAPI
- **Professional code structure** following best practices

## 📁 Project Structure

```
cats-vs-dogs-classifier/
├── src/
│   ├── models/          # Model architecture
│   ├── data/            # Data loading and preprocessing
│   ├── api/             # FastAPI application
│   ├── utils/           # Utility functions
│   ├── train.py         # Training script
│   ├── evaluate.py      # Evaluation script
│   ├── predict.py       # Inference script
│   └── config.py        # Configuration settings
├── data/
│   ├── raw/             # Raw dataset (cats/ and dogs/ folders)
│   └── processed/       # Processed data
├── models/
│   └── checkpoints/     # Saved model checkpoints
├── logs/                # TensorBoard logs
├── tests/               # Unit tests
├── notebooks/           # Jupyter notebooks for exploration
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## 🚀 Getting Started

### 1. Setup Environment

```bash
# Navigate to project directory
cd cats-vs-dogs-classifier

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download the Cats vs Dogs dataset and organize it as follows:

```
data/raw/
├── cats/
│   ├── cat.1.jpg
│   ├── cat.2.jpg
│   └── ...
└── dogs/
    ├── dog.1.jpg
    ├── dog.2.jpg
    └── ...
```

**Dataset sources:**
- [Kaggle Cats vs Dogs](https://www.kaggle.com/c/dogs-vs-cats/data)
- [Microsoft Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765)

### 3. Train the Model

```bash
# Train with default settings
python src/train.py

# Monitor training with TensorBoard
tensorboard --logdir=logs
```

**Training features:**
- Automatic train/val/test split (80/10/10)
- Data augmentation (rotation, flip, color jitter)
- Learning rate scheduling
- Checkpoint saving (best and periodic)
- TensorBoard logging

### 4. Evaluate the Model

```bash
# Evaluate on test set
python src/evaluate.py
```

This will:
- Calculate accuracy and metrics
- Generate confusion matrix
- Visualize predictions with confidence scores

### 5. Make Predictions

```bash
# Use the predictor in your code
from src.predict import Predictor
from pathlib import Path

predictor = Predictor("models/checkpoints/best_model.pth")
predicted_class, confidence = predictor.predict(Path("path/to/image.jpg"))
print(f"Prediction: {predicted_class} ({confidence:.2%} confidence)")
```

### 6. Deploy API

```bash
# Start FastAPI server
python src/api/app.py

# Or use uvicorn directly
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

**API Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `GET /info` - Model information
- `POST /predict` - Upload image for classification

**Example API usage:**

```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@cat.jpg"

# Response
{
  "prediction": "cat",
  "confidence": 0.9876,
  "all_classes": {
    "cat": 0.9876,
    "dog": 0.0124
  }
}
```

## 🧠 Model Architecture

- **Backbone:** ResNet50 (pre-trained on ImageNet)
- **Transfer Learning:** Frozen backbone + custom classifier
- **Classifier:** 
  - Dropout(0.5)
  - Linear(2048 → 512)
  - ReLU
  - Dropout(0.3)
  - Linear(512 → 2)

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Model
MODEL_NAME = "resnet50"
NUM_CLASSES = 2

# Training
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Image
IMAGE_SIZE = 224
```

## 📊 Expected Results

With the default configuration, you should achieve:
- **Training Accuracy:** ~95-98%
- **Validation Accuracy:** ~90-95%
- **Test Accuracy:** ~90-95%

## 🔧 Advanced Usage

### Fine-tuning

To fine-tune the entire model:

```python
from src.models.model import create_model

model = create_model(freeze_backbone=False)
# Train with lower learning rate
```

### Custom Dataset

To use your own dataset:
1. Organize images in `data/raw/class1/` and `data/raw/class2/`
2. Update `CLASS_NAMES` in `src/config.py`
3. Update `NUM_CLASSES` if needed

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

## 📝 Key Concepts Learned

This project teaches:

1. **Transfer Learning:** Using pre-trained models
2. **Data Augmentation:** Improving generalization
3. **PyTorch Basics:** Datasets, DataLoaders, Models
4. **Training Loop:** Forward/backward pass, optimization
5. **Model Evaluation:** Metrics, confusion matrix
6. **API Deployment:** FastAPI for ML models
7. **Best Practices:** Modular code, logging, configuration

## 🎓 Next Steps

After mastering this project, try:

1. **Add more architectures:** EfficientNet, Vision Transformer
2. **Implement TensorFlow version:** Compare PyTorch vs TensorFlow
3. **Add more classes:** Multi-class classification
4. **Deploy to cloud:** AWS, GCP, or Azure
5. **Add Docker:** Containerize the application
6. **Implement CI/CD:** Automated testing and deployment

## 📚 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

## 🤝 Contributing

Feel free to:
- Add new features
- Improve documentation
- Report bugs
- Suggest enhancements

## 📄 License

MIT License - feel free to use this project for learning and development.

---

**Happy Learning! 🚀**

*Built with ❤️ using PyTorch and FastAPI*
