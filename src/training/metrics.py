import torch


def accuracy(outputs, targets):
    """Top-1 accuracy for a batch."""
    _, predicted = torch.max(outputs, 1)
    correct = predicted.eq(targets).sum().item()
    total = targets.size(0)
    return correct, total


def top_k_accuracy(outputs, targets, k=5):
    """Top-k accuracy for a batch."""
    _, top_k_preds = torch.topk(outputs, k, dim=1)
    targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)
    correct = top_k_preds.eq(targets_expanded).any(dim=1).sum().item()
    total = targets.size(0)
    return correct, total


def collect_predictions(model, dataloader, device):
    """
    Run model over entire dataloader and collect all true labels
    and predicted labels. Used for confusion matrix and per-class accuracy.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return all_targets, all_preds