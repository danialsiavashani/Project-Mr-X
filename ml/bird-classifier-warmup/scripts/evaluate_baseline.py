import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.device import get_device
from src.utils.paths import VAL_DIR, CHECKPOINTS_DIR, PLOTS_DIR
from src.utils.config import NUM_CLASSES, BATCH_SIZE, CLASS_NAMES
from src.datasets.transforms import get_val_transforms
from src.datasets.bird_dataset import get_dataset
from src.models.baseline_cnn import BaselineCNN
from src.training.metrics import collect_predictions


def main():
    device = get_device()

    # data
    val_dataset = get_dataset(VAL_DIR, get_val_transforms())
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # load model
    model = BaselineCNN(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(CHECKPOINTS_DIR / 'baseline_cnn_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint — epoch {checkpoint['epoch']}, val acc {checkpoint['val_acc']:.2f}%\n")

    # collect predictions
    true_labels, pred_labels = collect_predictions(model, val_loader, device)

    # confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix — BaselineCNN (val set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'confusion_matrix_baseline.png', dpi=150)
    print(f"Confusion matrix saved to outputs/plots/")

    # per class report
    print("\nPer-class report:")
    print(classification_report(true_labels, pred_labels, target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()