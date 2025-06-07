import torch
from torch.utils.data import Dataset
import zarr
import numpy as np

class ZarrDataset(Dataset):
    def __init__(self, x_path, y_path, batch_size):
        self.X = zarr.open(x_path, mode='r')
        self.Y = zarr.open(y_path, mode='r')
        self.batch_size = batch_size
        self.indices = np.arange(self.X.shape[0] // batch_size)
        np.random.shuffle(self.indices)

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        i = idx * self.batch_size
        j = i + self.batch_size
        x = self.X[i:j]  # liest Blockweise = effizient
        y = self.Y[i:j]
        return torch.from_numpy(x), torch.from_numpy(y).long()
