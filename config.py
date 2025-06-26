import torch
from pathlib import Path


working_dir = Path(__file__).parent.resolve()

config = {

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": working_dir / "checkpoints",
    "resume_path": None,
    "model_version": 0,

    ## training

    "batch_size": 524,
    "lr": 0.00025,
    "epochs": 1000,
    "value_weight": 1,
    "policy_weight": 1,
    "weight_decay": 0.0001,
    "batch_log": 100,

    ## training_data
    "num_workers": 5,
    "path_X": working_dir / "data/new_stockfish_processed/2540000/boards",
    "path_Y": working_dir / "data/new_stockfish_processed/2540000/values",
    "chunks_per_set": 4,
    "prefetch_per_worker": 2,

    ## persistent data

    "data_path": working_dir / "data/raw/tactic_evals.csv",

    ## eval
    "eval_model": working_dir / "checkpoints/model_15.pt",
    "dataset_X": working_dir / "data/stockfish/copy/boards",
    "dataset_Y": working_dir / "data/stockfish/copy/moves",
    "eval_batch_size": 1024,
    "eval_log": 20
}
