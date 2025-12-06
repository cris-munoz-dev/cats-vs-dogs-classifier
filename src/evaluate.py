"""Evaluation script for the Cats vs Dogs classifier."""

import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
import logging

from src.models.model import create_model
from src.data.dataset import create_dataloaders
from src.config import CHECKPOINTS_DIR, CLASS_NAMES, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator class for model evaluation and visualization."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader,
        device: str = "cpu",
        class_names: list = CLASS_NAMES
    ):
        """
        Args:
            model: Trained PyTorch model
            test_loader: Test data loader
            device: Device to run evaluation on
            class_names: List of class names
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.model.eval()
    
    def evaluate(self):
        """Evaluate the model on test data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        all_labels = []
        correct = 0
        total = 0
        
        logger.info("Evaluating model...")
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        
        # Generate classification report
        report = classification_report(
            all_labels,
            all_predictions,
            target_names=self.class_names,
            output_dict=True
        )
        
        logger.info(f"\nTest Accuracy: {accuracy:.2f}%")
        logger.info("\nClassification Report:")
        logger.info(classification_report(
            all_labels,
            all_predictions,
            target_names=self.class_names
        ))
        
        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'report': report
        }
    
    def plot_confusion_matrix(
        self,
        predictions: list,
        labels: list,
        save_path: Path = None
    ):
        """Plot confusion matrix.
        
        Args:
            predictions: List of predicted labels
            labels: List of true labels
            save_path: Optional path to save the plot
        """
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def visualize_predictions(
        self,
        num_images: int = 16,
        save_path: Path = None
    ):
        """Visualize model predictions on sample images.
        
        Args:
            num_images: Number of images to visualize
            save_path: Optional path to save the plot
        """
        self.model.eval()
        
        # Get a batch of images
        images, labels = next(iter(self.test_loader))
        images, labels = images[:num_images], labels[:num_images]
        images, labels = images.to(self.device), labels.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
        
        # Plot
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        axes = axes.ravel()
        
        for idx in range(num_images):
            # Denormalize image
            img = images[idx].cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0).numpy()
            
            # Get prediction info
            true_label = self.class_names[labels[idx]]
            pred_label = self.class_names[predicted[idx]]
            confidence = probabilities[idx][predicted[idx]].item() * 100
            
            # Plot
            axes[idx].imshow(img)
            axes[idx].axis('off')
            
            color = 'green' if true_label == pred_label else 'red'
            axes[idx].set_title(
                f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                color=color,
                fontsize=10
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions visualization saved to {save_path}")
        
        plt.show()


def main():
    """Main evaluation function."""
    # Load test data
    logger.info("Loading test data...")
    _, _, test_loader = create_dataloaders(RAW_DATA_DIR)
    
    # Load model
    logger.info("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device=device)
    
    # Load best checkpoint
    checkpoint_path = CHECKPOINTS_DIR / "best_model.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        logger.warning("No checkpoint found! Using untrained model.")
    
    # Evaluate
    evaluator = Evaluator(model, test_loader, device)
    results = evaluator.evaluate()
    
    # Visualize
    evaluator.plot_confusion_matrix(
        results['predictions'],
        results['labels'],
        save_path=Path("confusion_matrix.png")
    )
    
    evaluator.visualize_predictions(
        num_images=16,
        save_path=Path("predictions.png")
    )


if __name__ == "__main__":
    main()
