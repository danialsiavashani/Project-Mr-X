import torch


def predict(model, image_tensor, class_names, device, top_k=3):
    """
    Run inference on a single preprocessed image tensor.
    Returns top-k predictions as (class_name, confidence%) tuples.
    """
    model.eval()
    tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k)

    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        results.append((class_names[idx.item()], prob.item() * 100))
    return results