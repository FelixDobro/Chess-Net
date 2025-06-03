import torch
import torch.nn.functional as F
import gc
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import config
from data.chunked_dataset import ChunkedDataset

def train(model, chunk_files, logger):
    device = config["device"]
    print("Device: ", device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    chunks_per_set = config["chunks_per_set"]
    partitions = [chunk_files[i * chunks_per_set: (i + 1) * chunks_per_set]
    for i in range(len(chunk_files) // chunks_per_set)]

    scaler = GradScaler(device)

    for epoch in range(config["epochs"]):
        total_loss = 0.0
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
                xb, p_targets = xb.to(device), p_targets.to(device)
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
                        "step/loss_policy": loss.item(),
                        "step": step
                    })
            del loader
            del dataset
            gc.collect()


        avg_loss = total_loss / total_samples
        logger.log({
            "train/loss_policy": avg_loss,
            "train/lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch
        })
        torch.save(model.state_dict(), f"{config['model_path']}/model_{config['model_version']}.pt")
        config["model_version"] += 1

    logger.finish()
