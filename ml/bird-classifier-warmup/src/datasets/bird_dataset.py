from torchvision.datasets import ImageFolder


def get_dataset(split_dir, transform):
    """
    Loads a dataset from a directory using ImageFolder.
    Expects split_dir to contain one subfolder per class.

    Args:
        split_dir: path to train/, valid/, or test/ folder
        transform: transform pipeline to apply

    Returns:
        ImageFolder dataset
    """
    return ImageFolder(root=split_dir, transform=transform)