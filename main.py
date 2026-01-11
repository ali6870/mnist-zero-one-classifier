import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12
EPOCHS = 3
LR = 1E-3
PIXEL_THRESH = 0.5

transform = transforms.ToTensor()

train_full = datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_full = datasets.MNIST(root='data', train=True, transform=transform, download=True)

def filter_zero_ones(ds):
    idx = [i for i, (_,y) in enumerate(ds) if y in (0, 1)]
    return Subset(ds, idx)

train_ds = filter_zero_ones(train_full)
test_ds = filter_zero_ones(test_full)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

class BinaryMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # logit
        )

    def forward(self, x):
        return self.net(x)

model = BinaryMNIST().to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=LR)


def binarize_pixels(x):
    return (x >= PIXEL_THRESH).float()

@torch.no_grad()
def accuracy(loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = binarize_pixels(x).to(DEVICE)
        y = y.float().to(DEVICE)

        logits = model(x).squeeze(1)
        preds = (torch.sigmoid(logits) >= 0.5).float()

        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / total

for epoch in range(1, EPOCHS + 1):
    model.train()
    for x, y in train_loader:
        x = binarize_pixels(x).to(DEVICE)
        y = y.float().to(DEVICE)

        logits = model(x).squeeze(1)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(
        f"Epoch {epoch} | "
        f"Train acc: {accuracy(train_loader):.4f} | "
        f"Test acc: {accuracy(test_loader):.4f}"
    )


@torch.no_grad()
def show_predictions(n=10):
    model.eval()
    x, y = next(iter(test_loader))
    x = binarize_pixels(x).to(DEVICE)

    probs = torch.sigmoid(model(x).squeeze(1)).cpu()
    preds = (probs >= 0.5).int()

    for i in range(n):
        print(
            f"true={int(y[i])}  "
            f"pred={int(preds[i])}  "
            f"prob_1={probs[i]:.3f}"
        )

print("\nPredictions:")
show_predictions(12)