import torch
from torch import autocast

from config import config
from models.chessnet import ChessNet
from torch.utils.data import DataLoader
from data.zarr_dataset import ZarrDataset

def main():
    model = ChessNet()
    model_route = config["eval_model"]
    model.load_state_dict(torch.load(model_route, map_location=config["device"]))
    model = model.to(config["device"])
    num_accurate_preds = 0
    total_samples = 0
    model.eval()
    dataset = ZarrDataset(config["dataset_X"], config["dataset_Y"], config["eval_batch_size"])
    loader = DataLoader(dataset, 1, num_workers=config["num_workers"])

    for step, (xb, p_targets) in enumerate(loader):
        with torch.no_grad():

            xb = xb.squeeze(0).to(config["device"])
            p_targets = p_targets.to(config["device"])
            y_pred = model(xb)
        batch_size = xb.size(0)
        corrects = torch.argmax(y_pred, dim=-1) == p_targets
        num_accurate_preds += torch.sum(corrects)
        total_samples += batch_size
        if (total_samples / batch_size) % config["eval_log"] == 0:
            print(total_samples)
            print(f"Accuracy: {num_accurate_preds / total_samples}")

    print(f"Accuracy: {num_accurate_preds / total_samples}")

if __name__ == '__main__':
    main()