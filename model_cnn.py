import torch.nn as nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)  # 32x32 -> 32x32
        self.c2 = nn.Conv2d(16, 32, 3, padding=1) # 16x16 -> 16x16
        self.fc1 = nn.Linear(32*8*8, 64)
        self.fc2 = nn.Linear(64, 2)               # 2 classes: circle / not_circle

    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.max_pool2d(x, 2)  # -> 16x16
        x = F.relu(self.c2(x)); x = F.max_pool2d(x, 2)  # -> 8x8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits
