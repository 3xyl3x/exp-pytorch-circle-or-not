# check_loader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter

def main():
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # ensure 1 channel
        transforms.ToTensor(),                        # HxW -> tensor [1,H,W] in [0,1]
    ])

    train_ds = datasets.ImageFolder("data/train", transform=tfm)
    val_ds   = datasets.ImageFolder("data/val",   transform=tfm)

    print("Class -> index mapping:", train_ds.class_to_idx)  # e.g. {'circle': 0, 'not_circle': 1}

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    # Grab one batch to verify shapes & labels
    x, y = next(iter(train_loader))
    print("Batch tensor shape:", x.shape)   # expect [B, 1, 32, 32]
    print("Batch labels:", y.tolist())

    # Quick class counts in the whole train set
    inv = {v:k for k,v in train_ds.class_to_idx.items()}
    counts = Counter(lbl for _, lbl in train_ds.samples)
    print("Train counts per class:", {inv[k]: v for k, v in counts.items()})

if __name__ == "__main__":
    main()
