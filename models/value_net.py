import torch.nn.functional as F
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)


    def forward(self, x):
        out = self.relu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))
        out = self.relu(out + x)
        return out

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(18, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])


        self.value_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.res_blocks(x)
        v = self.value_head(x)
        return v
