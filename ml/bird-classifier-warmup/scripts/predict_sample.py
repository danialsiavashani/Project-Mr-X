import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.utils.device import get_device
from src.utils.paths import CHECKPOINTS_DIR, PREDICTIONS_DIR, IMAGES_TO_PREDICT_DIR
from src.utils.config import NUM_CLASSES, CLASS_NAMES
from src.datasets.transforms import get_val_transforms
from src.models.baseline_cnn import BaselineCNN
from src.training.predict import predict

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    """Convert normalized tensor back to viewable image."""
    img = tensor.numpy().transpose(1, 2, 0)
    img = STD * img + MEAN
    img = np.clip(img, 0, 1)
    return img


def main():
    device = get_device()
    transform = get_val_transforms()

    # load model
    model = BaselineCNN(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(CHECKPOINTS_DIR / 'baseline_cnn_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint — epoch {checkpoint['epoch']}, val acc {checkpoint['val_acc']:.2f}%\n")

    # load images
    image_files = sorted([f for f in os.listdir(IMAGES_TO_PREDICT_DIR) if f.endswith('.jpg')])

    n = len(image_files)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for i, fname in enumerate(image_files):
        image_path = os.path.join(IMAGES_TO_PREDICT_DIR, fname)
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image)

        results = predict(model, tensor, CLASS_NAMES, device)
        top_name, top_conf = results[0]

        img_display = denormalize(tensor)
        axes[i].imshow(img_display)
        axes[i].set_title(
            f"{fname}\n{top_name}\n{top_conf:.1f}%",
            fontsize=8
        )
        axes[i].axis('off')

        print(f"Image: {fname}")
        for j, (name, prob) in enumerate(results):
            print(f"  Top {j+1}: {name:<35} {prob:.2f}%")
        print()

    plt.suptitle('Sample Predictions — BaselineCNN', fontsize=13)
    plt.tight_layout()
    plt.savefig(PREDICTIONS_DIR / 'predict_sample_baseline.png', dpi=150)
    print(f"Saved to outputs/predictions/")


if __name__ == "__main__":
    main()