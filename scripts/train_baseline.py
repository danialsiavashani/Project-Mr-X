import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import time
from src.utils.device import get_device
from src.utils.seed import set_seed
from src.utils.paths import TRAIN_DIR, VAL_DIR, CHECKPOINTS_DIR
from src.utils.config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, NUM_CLASSES, SEED
from src.datasets.transforms import get_train_transforms, get_val_transforms
from src.datasets.bird_dataset import get_dataset
from src.models.baseline_cnn import BaselineCNN
from src.training.train import train_one_epoch
from src.training.evaluate import evaluate


def main():
    # setup
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    # data
    train_dataset = get_dataset(TRAIN_DIR, get_train_transforms())
    val_dataset   = get_dataset(VAL_DIR,   get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Classes : {train_dataset.classes}")
    print(f"Train   : {len(train_dataset)} samples")
    print(f"Val     : {len(val_dataset)} samples")

    # model
    model = BaselineCNN(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop
    start_time = time.time()
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%"
        )
        elapsed = time.time() - start_time
        print(f"  Time elapsed: {elapsed:.0f}s")

        # save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = CHECKPOINTS_DIR / "baseline_cnn_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"  ✓ Saved best checkpoint (val acc: {val_acc:.2f}%)")

    total_time = time.time() - start_time
    print(f"\nDone. Best val acc: {best_val_acc:.2f}%")
    print(f"Total training time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"Checkpoint saved to: {CHECKPOINTS_DIR / 'baseline_cnn_best.pth'}")


if __name__ == "__main__":
    main()