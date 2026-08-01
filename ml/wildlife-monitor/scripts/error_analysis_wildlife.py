import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from src.utils.device import get_device
from src.utils.paths import VAL_DIR, CHECKPOINTS_DIR, PLOTS_DIR
from src.utils.config import NUM_CLASSES, BATCH_SIZE, CLASS_NAMES
from src.datasets.transforms import get_val_transforms
from src.datasets.wildlife_dataset import get_dataset
from src.models.efficientnet_classifier import EfficientNetClassifier


def main():
    device = get_device()

    val_dataset = get_dataset(str(VAL_DIR), get_val_transforms())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = EfficientNetClassifier(num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(CHECKPOINTS_DIR / "efficientnet_wildlife_best.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    file_paths = [s[0] for s in val_dataset.samples]
    errors = [
        (file_paths[i], all_targets[i], all_preds[i])
        for i in range(len(all_targets))
        if all_targets[i] != all_preds[i]
    ]

    print(f"Total errors: {len(errors)} / {len(all_targets)}")
    print(f"Error rate: {len(errors)/len(all_targets)*100:.1f}%\n")

    n = min(12, len(errors))
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for i in range(n):
        path, true, pred = errors[i]
        img = Image.open(path).convert('RGB')
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {CLASS_NAMES[true]}\nPred: {CLASS_NAMES[pred]}",
            fontsize=7, color='red'
        )
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Misclassified Images — Wildlife EfficientNet', fontsize=12)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(PLOTS_DIR / 'error_analysis_wildlife.png', dpi=150)
    print(f"Error analysis saved to {PLOTS_DIR / 'error_analysis_wildlife.png'}")


if __name__ == "__main__":
    main()