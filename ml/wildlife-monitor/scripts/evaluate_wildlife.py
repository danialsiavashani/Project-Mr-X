import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.utils.device import get_device
from src.utils.paths import VAL_DIR, CHECKPOINTS_DIR, PLOTS_DIR
from src.utils.config import NUM_CLASSES, BATCH_SIZE, CLASS_NAMES
from src.datasets.transforms import get_val_transforms
from src.datasets.wildlife_dataset import get_dataset
from src.models.efficientnet_classifier import EfficientNetClassifier
from src.training.metrics import collect_predictions


def main():
    device = get_device()

    val_dataset = get_dataset(str(VAL_DIR), get_val_transforms())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = EfficientNetClassifier(num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(CHECKPOINTS_DIR / "efficientnet_wildlife_best.pth", map_location=device)
    model.load_state_dict(state_dict)  # bare state_dict, matches how it was actually saved tonight

    true_labels, pred_labels = collect_predictions(model, val_loader, device)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix — Wildlife EfficientNet (val set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'confusion_matrix_wildlife.png', dpi=150)
    print(f"Confusion matrix saved to {PLOTS_DIR / 'confusion_matrix_wildlife.png'}")

    print("\nPer-class report:")
    print(classification_report(true_labels, pred_labels, target_names=CLASS_NAMES, digits=3))


if __name__ == "__main__":
    main()