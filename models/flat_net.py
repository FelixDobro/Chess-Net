import torch
from torch import nn
from torch.nn import functional as F


# Converts a chess.Board object into a flat 773-dimensional vector:
# - 768 bits: one-hot encoding of piece type and position (12 types × 64 squares)
# - 5 additional features: side to move and castling rights

# A simple feedforward network for evaluating chess positions.
# Input: 773-dim vector from board encoding
# Output: single scalar evaluation (e.g., approximating Stockfish score)

class FlatNet(nn.Module):
    def __init__(self):
        super(FlatNet, self).__init__()
        self.input_layer = nn.Linear(12*64+5, 512)
        self.hidden_layer_1 = nn.Linear(512, 256)
        self.hidden_layer_2 = nn.Linear(256, 32)
        self.hidden_layer_3 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer_1(x))
        x = F.relu(self.hidden_layer_2(x))
        x = F.relu(self.hidden_layer_3(x))
        x = self.output_layer(x)
        return x

