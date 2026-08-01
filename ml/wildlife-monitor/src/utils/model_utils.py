def freeze_backbone(model):
    """Freeze all layers except the classifier head."""
    for name, param in model.backbone.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False


def unfreeze_backbone(model):
    """Unfreeze all layers for Stage 2 fine-tuning."""
    for param in model.backbone.parameters():
        param.requires_grad = True


def count_trainable_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())