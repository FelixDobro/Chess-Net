
import torch


from config import config


from training.train_value import train
from models.value_net import ChessNet
from models.flat_net import FlatNet

from utils.logger import Logger

if __name__ == "__main__":
    logger = Logger(project="schachbot", config=config)
    model = FlatNet().to(config["device"])
    if config["resume_path"]:
        model.load_state_dict(torch.load(config["resume_path"]))
    logger.watch_model(model)

    train(
        model=model,
        logger=logger,
    )

    logger.finish()

