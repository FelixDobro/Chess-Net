import torch

config = {

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "checkpoints",
    "resume_path": None,
    "epoch": 0.1,

    ## training

    "batch_size": 64,
    "lr": 0.0001,
    "epochs": 1000,
    "log_freq": 200,
    "value_weight": 1,
    "policy_weight": 1,
    "weight_decay": 0.0001


    ## data
    "num_workers": 25,
    "data_path": "data/training_chunk_data",
    "chunks_per_epoch": 221,
    "prefetch_per_worker": 6
}
