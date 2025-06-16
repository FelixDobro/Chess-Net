import torch
from torch.utils.data import Dataset
import zarr
import numpy as np

class ZarrDataset(Dataset):
    def __init__(self, x_path, y_path, batch_size):
        self.X = zarr.open(x_path, mode='r')
        self.Y = zarr.open(y_path, mode='r')
        self.batch_size = batch_size
        self.n_batches = self.X.shape[0] // self.batch_size + (1 if self.X.shape[0] % self.batch_size != 0 else 0)

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.X.shape[0])
        x = self.X[start:end]
        y = self.Y[start:end]
        return torch.from_numpy(x).float(), torch.from_numpy(y).long()
