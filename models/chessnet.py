import torch.nn.functional as F
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + x)
        return out

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(18, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.shared_res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(25)])

        # Split Heads
        # Policy Tower
        self.policy_head = nn.Sequential(
            *[ResidualBlock(256) for _ in range(5)],
            nn.Conv2d(256, 128, kernel_size=1),  # 1x1 conv für Komprimierung
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1968)
        )

        # Value Tower
        self.value_head = nn.Sequential(
            *[ResidualBlock(256) for _ in range(5)],
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.shared_res_blocks(x)

        p = self.policy_head(x)
        v = self.value_head(x)

        return p, v
