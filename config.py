import torch

config = {

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "checkpoints",
    "resume_path": "checkpoints/model_4.pt",
    "model_version": 5,

    ## training

    "batch_size": 1024,
    "lr": 0.0001,
    "epochs": 1000,
    "value_weight": 1,
    "policy_weight": 1,
    "weight_decay": 0.0001,
    "batch_log": 20,

    ## data
    "num_workers": 0,
    "data_path": "data/new_data",
    "chunks_per_set": 4,
    "prefetch_per_worker": 2
}
