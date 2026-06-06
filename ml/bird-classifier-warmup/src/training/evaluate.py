import torch
from src.training.metrics import accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Run evaluation on a dataloader.
    Returns average loss and accuracy.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * data.size(0)
            correct, count = accuracy(outputs, targets)
            total_correct += correct
            total_samples += count

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples

    return avg_loss, avg_acc