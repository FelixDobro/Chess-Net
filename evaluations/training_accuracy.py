import gc
import glob
from pathlib import Path
import torch
import numpy as np
import os

from torch.utils.data import DataLoader

from config import config
from models.chessnet import ChessNet
from data.chunked_dataset import ChunkedDataset

def main():
    model = ChessNet()
    model_route = config["content_root"] / config["selected_model"]
    model.load_state_dict(torch.load(model_route, map_location=config["device"]))
    model.eval()
    data_route = config["content_root"] / config["data_path"]
    chunk_files = glob.glob(f'{data_route}/*.npz')
    chunks_per_set = config["chunks_per_set"]
    partitions = [chunk_files[i * chunks_per_set: (i + 1) * chunks_per_set]
                  for i in range(len(chunk_files) // chunks_per_set)]
    num_accurate_preds = 0
    total_samples = 0
    for partition in partitions:
        dataset = ChunkedDataset(partition)
        loader = DataLoader(
            dataset=dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True
        )
        model.train()
        for step, (xb, p_targets) in enumerate(loader):
            with torch.no_grad():
                y_pred = model(xb)

            batch_size = xb.size(0)
            corrects = torch.argmax(y_pred, dim=-1) == p_targets
            num_accurate_preds += torch.sum(corrects)
            total_samples += batch_size
            if (total_samples / batch_size) % 10 == 0:
                print(total_samples)
                print(f"Accuracy: {num_accurate_preds / total_samples}")
        del loader
        del dataset
        gc.collect()

    print(f"Accuracy: {num_accurate_preds / total_samples}")

if __name__ == '__main__':
    main()