import torch
from pathlib import Path


working_dir = Path(__file__).parent.resolve()

config = {

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": working_dir / "checkpoints",
    "resume_path": working_dir / "checkpoints/model_2.pt",
    "model_version": 3,

    ## training

    "batch_size": 1024,
    "lr": 0.0002,
    "epochs": 1000,
    "value_weight": 1,
    "policy_weight": 1,
    "weight_decay": 0.0001,
    "batch_log": 20,

    ## data
    "num_workers": 5,
    "path_X": working_dir / "data/new_stockfish_processed/2540000/boards",
    "path_Y": working_dir / "data/new_stockfish_processed/2540000/values",
    "chunks_per_set": 4,
    "prefetch_per_worker": 2,

    ## eval
    "eval_model": working_dir / "checkpoints/model_15.pt",
    "dataset_X": working_dir / "data/stockfish/copy/boards",
    "dataset_Y": working_dir / "data/stockfish/copy/moves",
    "eval_batch_size": 1024,
    "eval_log": 20
}
