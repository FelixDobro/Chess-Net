import torch

config = {

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "checkpoints",
    "resume_path": None,
    "epoch": 1,

    ## training

    "batch_size": 64,
    "lr": 0.001,
    "epochs": 1000,
    "log_freq": 20,
    "value_weight": 1,
    "policy_weight": 1,


    ## data
    "num_workers": 1,
    "data_path": "data/training_chunk_data",
    "chunks_per_epoch": 10,
    "prefetch_per_worker": 10
}
