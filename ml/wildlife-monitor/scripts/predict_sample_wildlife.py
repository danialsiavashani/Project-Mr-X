import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.utils.device import get_device
from src.utils.paths import VAL_DIR, CHECKPOINTS_DIR, PREDICTIONS_DIR
from src.utils.config import NUM_CLASSES, CLASS_NAMES, SEED
from src.datasets.transforms import get_val_transforms
from src.datasets.wildlife_dataset import get_dataset
from src.models.efficientnet_classifier import EfficientNetClassifier

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    img = tensor.numpy().transpose(1, 2, 0)
    img = STD * img + MEAN
    return np.clip(img, 0, 1)


def main():
    random.seed(SEED)
    device = get_device()
    transform = get_val_transforms()

    model = EfficientNetClassifier(num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(CHECKPOINTS_DIR / "efficientnet_wildlife_best.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    val_dataset = get_dataset(str(VAL_DIR), transform)
    sample_indices = random.sample(range(len(val_dataset)), 6)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for i, idx in enumerate(sample_indices):
        tensor, true_label = val_dataset[idx]
        file_path = val_dataset.samples[idx][0]

        with torch.no_grad():
            outputs = model(tensor.unsqueeze(0).to(device))
            probs = torch.softmax(outputs, dim=1)[0]
            top_probs, top_idxs = torch.topk(probs, 3)

        results = [
            (CLASS_NAMES[idx.item()], min(prob.item(), 0.99) * 100)
            for idx, prob in zip(top_idxs, top_probs)
        ]
        top_name, top_conf = results[0]
        correct = (top_idxs[0].item() == true_label)

        img_display = denormalize(tensor)
        axes[i].imshow(img_display)
        title_color = 'green' if correct else 'red'
        axes[i].set_title(
            f"True: {CLASS_NAMES[true_label]}\nPred: {top_name} ({top_conf:.1f}%)",
            fontsize=8, color=title_color
        )
        axes[i].axis('off')

        print(f"Image: {os.path.basename(file_path)}  (true: {CLASS_NAMES[true_label]})")
        for name, prob in results:
            print(f"  {name:<25} {prob:.1f}%")
        print()

    plt.suptitle('Sample Predictions — Wildlife EfficientNet', fontsize=13)
    plt.tight_layout()
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    plt.savefig(PREDICTIONS_DIR / 'predict_sample_wildlife.png', dpi=150)
    print(f"Saved to {PREDICTIONS_DIR / 'predict_sample_wildlife.png'}")


if __name__ == "__main__":
    main()