import torch
from src.training.metrics import accuracy


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        correct, count = accuracy(outputs, targets)
        total_correct += correct
        total_samples += count

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples

    return avg_loss, avg_acc