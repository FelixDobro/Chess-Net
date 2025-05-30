import torch

config = {
    "api_key": "475b6fdb3b2879cce8dd4d1b90912b4c0abe1d63",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "checkpoints",
    "resume_path": None,
    "epoch": 0.1,

    ## training

    "batch_size": 64,
    "lr": 0.001,
    "epochs": 10,
    "log_freq": 20,
    "value_weight": 0.9,
    "policy_weight": 1,


    ## data
    "num_workers": 8,
    "data_path": "data/training_chunk_data",
    "chunks_per_epoch": 1,
    "prefetch_per_worker": 4
}
