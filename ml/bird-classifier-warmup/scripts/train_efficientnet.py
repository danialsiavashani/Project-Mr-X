import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import time

from src.utils.device import get_device
from src.utils.seed import set_seed
from src.utils.paths import TRAIN_DIR, VAL_DIR, CHECKPOINTS_DIR
from src.utils.config import (BATCH_SIZE, NUM_CLASSES, SEED,
                               EFFICIENTNET_STAGE1_EPOCHS,
                               EFFICIENTNET_STAGE2_EPOCHS,
                               EFFICIENTNET_STAGE1_LR,
                               EFFICIENTNET_STAGE2_LR)
from src.utils.model_utils import freeze_backbone, unfreeze_backbone, count_trainable_params, count_total_params
from src.datasets.transforms import get_train_transforms, get_val_transforms
from src.datasets.bird_dataset import get_dataset
from src.models.efficientnet_classifier import EfficientNetClassifier
from src.training.train import train_one_epoch
from src.training.evaluate import evaluate


def run_stage(model, train_loader, val_loader, criterion, optimizer,
              num_epochs, device, best_val_acc, stage_name, start_time):
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        print(
            f"[{stage_name}] Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
            f"Time: {elapsed:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = CHECKPOINTS_DIR / "efficientnet_best.pth"
            torch.save({
                'epoch': epoch,
                'stage': stage_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"  ✓ Saved best checkpoint (val acc: {val_acc:.2f}%)")

    return best_val_acc


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    # data
    train_dataset = get_dataset(TRAIN_DIR, get_train_transforms())
    val_dataset   = get_dataset(VAL_DIR,   get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_dataset)} samples  Val: {len(val_dataset)} samples")

    # model
    model = EfficientNetClassifier(num_classes=NUM_CLASSES).to(device)
    freeze_backbone(model)

    print(f"Total params:     {count_total_params(model):,}")
    print(f"Trainable params: {count_trainable_params(model):,}  (Stage 1 — head only)\n")

    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    best_val_acc = 0.0

    # stage 1 — frozen backbone, train head only
    print("=== Stage 1 — Frozen backbone ===")
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=EFFICIENTNET_STAGE1_LR
    )
    best_val_acc = run_stage(model, train_loader, val_loader, criterion, optimizer,
                             EFFICIENTNET_STAGE1_EPOCHS, device, best_val_acc, "Stage1", start_time)

    # stage 2 — unfreeze, fine-tune whole network
    print(f"\n=== Stage 2 — Full fine-tuning ===")
    unfreeze_backbone(model)
    print(f"Trainable params: {count_trainable_params(model):,}  (Stage 2 — full network)\n")

    optimizer = optim.Adam(model.parameters(), lr=EFFICIENTNET_STAGE2_LR)
    best_val_acc = run_stage(model, train_loader, val_loader, criterion, optimizer,
                             EFFICIENTNET_STAGE2_EPOCHS, device, best_val_acc, "Stage2", start_time)

    total_time = time.time() - start_time
    print(f"\nDone. Best val acc: {best_val_acc:.2f}%")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Checkpoint: {CHECKPOINTS_DIR / 'efficientnet_best.pth'}")


if __name__ == "__main__":
    main()