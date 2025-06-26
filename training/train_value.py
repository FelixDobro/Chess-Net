import torch.nn.functional as F
import torch

from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import config
import webdataset as wds


def train(model, logger):
    device = config["device"]
    print("Device: ", device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    dataset = (wds.WebDataset("data/20M-unchanged/shard-{000000..001557}.tar.gz")
               .shuffle(10000)
               .decode("torch")
               .to_tuple("input.pth", "target.pth")
               )
    loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=config["num_workers"],
                        prefetch_factor=config["prefetch_per_worker"])
    scaler = GradScaler(device)

    for epoch in range(config["epochs"]):
        total_loss = 0.0
        total_samples = 0
        model.train()
        for step, (xb, v_targets) in enumerate(loader):

            xb, v_targets = xb.to(device), v_targets.to(device)

            optimizer.zero_grad()
            with autocast(device_type=device):
                v_out = model(xb)
                loss = F.mse_loss(v_targets, v_out)



            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            if step % config["batch_log"] == 0:
                logger.log({
                    "step/loss": loss.item(),
                })



        avg_loss = total_loss / total_samples
        logger.log({
            "step/epoch_loss": avg_loss,
            "epoch": epoch
        })
        torch.save(model.state_dict(), f"{config['model_path']}/model_{config['model_version']}.pt")
        config["model_version"] += 1

    logger.finish()
