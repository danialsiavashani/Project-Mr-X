import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from src.utils.device import get_device
from src.utils.paths import VAL_DIR, CHECKPOINTS_DIR, PLOTS_DIR
from src.utils.config import NUM_CLASSES, BATCH_SIZE, CLASS_NAMES
from src.datasets.transforms import get_val_transforms
from src.datasets.bird_dataset import get_dataset
from src.models.baseline_cnn import BaselineCNN


def main():
    device = get_device()

    val_dataset = get_dataset(VAL_DIR, get_val_transforms())
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = BaselineCNN(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(CHECKPOINTS_DIR / 'baseline_cnn_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # collect predictions with file paths
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # find misclassified samples
    file_paths = [s[0] for s in val_dataset.samples]
    errors = [
        (file_paths[i], all_targets[i], all_preds[i])
        for i in range(len(all_targets))
        if all_targets[i] != all_preds[i]
    ]

    print(f"Total errors: {len(errors)} / {len(all_targets)}")
    print(f"Error rate: {len(errors)/len(all_targets)*100:.1f}%\n")

    # show first 12 errors
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

    plt.suptitle('Misclassified Images — BaselineCNN', fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'error_analysis_baseline.png', dpi=150)
    print(f"Error analysis saved to outputs/plots/")


if __name__ == "__main__":
    main()