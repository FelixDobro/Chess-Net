import wandb
from config import config

wandb.login(key=config["api_key"])

class Logger:
    def __init__(self, project, config):
        wandb.init(project=project, config=config)

    def log(self, metrics: dict, step=None):
        wandb.log(metrics, step=step)

    def watch_model(self, model):
        wandb.watch(model)

    def finish(self):
        wandb.finish()
