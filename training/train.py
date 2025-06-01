import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import config
from data.chunked_dataset import ChunkedDataset


def train(model, chunk_files, logger):

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    device = config["device"]
    scaler = GradScaler(device)
    log_freq = config["log_freq"]


    for epoch in range(config["epochs"]):
        dataset = ChunkedDataset(chunk_files)
        loader = DataLoader(
            num_workers=config["num_workers"],
            pin_memory=True,
            dataset=dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            prefetch_factor=config["prefetch_per_worker"])

        model.train()
        total_loss = total_policy_loss = total_value_loss = total_samples = 0
        for i, (xb, p_targets, value_targets) in enumerate(loader):
            xb, p_targets, value_targets = xb.to(device), p_targets.to(device), value_targets.to(device)
            optimizer.zero_grad()
            with autocast(device):
                p_out, v_out = model(xb)
                v_out = v_out.squeeze(-1)

                value_loss = F.mse_loss(v_out, value_targets)
                policy_loss = F.cross_entropy(p_out, p_targets)
                loss = config.get("value_weight", 1.0) * value_loss + config.get("policy_weight", 1.0) * policy_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_policy_loss += policy_loss.item() * batch_size
            total_value_loss += value_loss.item() * batch_size
            total_samples += batch_size
            if i % log_freq == 0:
                logger.log({
                    "batch/loss_total": loss.item(),
                    "batch/loss_policy": policy_loss.item(),
                    "batch/loss_value": value_loss.item(),
                    "batch/lr": optimizer.param_groups[0]["lr"]
                })


        avg_loss = total_loss / total_samples
        avg_policy_loss = total_policy_loss / total_samples
        avg_value_loss = total_value_loss / total_samples
        logger.log({
            "train/loss_total": avg_loss,
            "train/loss_policy": avg_policy_loss,
            "train/loss_value": avg_value_loss,
            "train/lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch
        })
        torch.save(model.state_dict(), f"{config['model_path']}/model_{config['epoch']}.pt")
        config["epoch"] += 1



    logger.finish()
