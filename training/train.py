import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import config
from data.zarr_dataset import ZarrDataset

def train(model, logger):
    device = config["device"]
    print("Device: ", device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    dataset = ZarrDataset(config["path_X"], config["path_Y"], config["batch_size"])
    loader = DataLoader(dataset, batch_size=1, num_workers=config["num_workers"], prefetch_factor=config["prefetch_per_worker"])
    scaler = GradScaler(device)

    for epoch in range(config["epochs"]):
        total_loss = 0.0
        total_samples = 0

        for step, (xb, p_targets) in enumerate(loader):
            xb, p_targets = xb.squeeze(0).to(device), p_targets.squeeze(0).to(device)
            optimizer.zero_grad()
            with autocast(device_type=device):
                p_out = model(xb)
                loss = F.cross_entropy(p_out, p_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            if step % config["batch_log"] == 0:
                logger.log({
                    "step/loss": loss.item(),
                    "step": step
                })



        avg_loss = total_loss / total_samples
        logger.log({
            "step/epoch_loss": avg_loss,
            "epoch": epoch
        })
        torch.save(model.state_dict(), f"{config['model_path']}/model_{config['model_version']}.pt")
        config["model_version"] += 1

    logger.finish()
