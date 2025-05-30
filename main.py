import glob

import torch
from torch.utils.data import DataLoader

from config import config
from data.chunked_dataset import ChunkedDataset

from training.train import train
from models.chessnet import ChessNet

from utils.logger import Logger

if __name__ == "__main__":
    logger = Logger(project="schachbot", config=config)
    model = ChessNet().to(config["device"])
    if config["resume_path"]:
        model.load_state_dict(torch.load(config["resume_path"]))
    logger.watch_model(model)
    batch_size = config["batch_size"]
    chunk_dir = config["data_path"]
    chunk_files = glob.glob(f'{chunk_dir}/*.npz')[0:2]
    print(f"# of chunk files: {len(chunk_files)}")

    train(
        model=model,
        chunk_files = chunk_files,
        logger=logger,
    )

    logger.finish()

