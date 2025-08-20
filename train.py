import torch, os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_cnn import TinyCNN

def make_loaders(root="data", batch=64):
    tfm = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=tfm)
    val_ds   = datasets.ImageFolder(os.path.join(root, "val"),   transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)
    return train_ds, val_ds, train_loader, val_loader

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.numel()
    return correct / max(1,total)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, train_loader, val_loader = make_loaders()

    model = TinyCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = 8
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)

        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {ep}: train_loss={(running/len(train_ds)):.4f}  val_acc={val_acc:.2%}")

    # Save checkpoint (weights + class names)
    os.makedirs("models", exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(),
         "classes": train_ds.classes},  # e.g. ['circle','not_circle']
        "models/circle_vs_not.pth"
    )
    print("Saved models/circle_vs_not.pth")

if __name__ == "__main__":
    main()
