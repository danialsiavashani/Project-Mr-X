import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from src.datasets.wildlife_dataset import get_dataset
from src.datasets.transforms import get_train_transforms, get_val_transforms
from src.models.efficientnet_classifier import EfficientNetClassifier
from src.training.train import train_one_epoch
from src.training.evaluate import evaluate

# ---- Config ----
NUM_CLASSES = 15
BATCH_SIZE = 32
STAGE1_EPOCHS = 5   # frozen backbone, head only
STAGE2_EPOCHS = 10  # unfrozen, low LR fine-tuning
STAGE1_LR = 1e-3
STAGE2_LR = 1e-4

TRAIN_DIR = "ml/wildlife-monitor/data/processed/train"
VALID_DIR = "ml/wildlife-monitor/data/processed/valid"
CHECKPOINT_PATH = "ml/wildlife-monitor/outputs/checkpoints/efficientnet_wildlife_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Data ----
train_dataset = get_dataset(TRAIN_DIR, get_train_transforms())
valid_dataset = get_dataset(VALID_DIR, get_val_transforms())

print(f"Classes: {train_dataset.classes}")
assert len(train_dataset.classes) == NUM_CLASSES, \
    f"Expected {NUM_CLASSES} classes, found {len(train_dataset.classes)}"

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---- Model ----
model = EfficientNetClassifier(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()

os.makedirs("ml/wildlife-monitor/outputs/checkpoints", exist_ok=True)
best_val_acc = 0.0

# ---- Stage 1: freeze backbone, train head only ----
print("\n=== Stage 1: frozen backbone ===")
for param in model.backbone.features.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=STAGE1_LR
)

for epoch in range(STAGE1_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
    print(f"[Stage1 Epoch {epoch+1}/{STAGE1_EPOCHS}] "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"  -> saved new best checkpoint ({val_acc:.2f}%)")

# ---- Stage 2: unfreeze everything, fine-tune with low LR ----
print("\n=== Stage 2: full network unfrozen ===")
for param in model.backbone.features.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=STAGE2_LR)

for epoch in range(STAGE2_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
    print(f"[Stage2 Epoch {epoch+1}/{STAGE2_EPOCHS}] "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"  -> saved new best checkpoint ({val_acc:.2f}%)")

print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
print(f"Checkpoint saved to: {CHECKPOINT_PATH}")