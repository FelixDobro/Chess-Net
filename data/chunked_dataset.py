import random
import numpy as np
from torch.utils.data import Dataset
import torch
from config import config


class ChunkedDataset(Dataset):

    def __init__(self, chunk_files):
        self.chunks = random.sample(chunk_files, config["chunks_per_epoch"])
        self.samples = []

        for chunk in self.chunks:
            data = np.load(chunk)

            boards = torch.from_numpy(data["boards"]).float()
            moves = torch.from_numpy(data["moves"]).long()
            values = torch.from_numpy(data["values"]).float()

            for i in range(boards.shape[0]):
                self.samples.append((boards[i], moves[i], values[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]