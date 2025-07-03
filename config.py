from dataclasses import dataclass

import torch


@dataclass
class Configuration:
    dataset_id: str = "zhiyingzou0202/empty_test" # "ariG23498/license-detection-paligemma"

    project_name: str = "gemma-3-4b-pt-object-detection-aug" # "SmolVLM-256M-Instruct-object-detection-aug"
    model_id: str = "google/gemma-3-4b-pt" # "HuggingFaceTB/SmolVLM-256M-Instruct"
    checkpoint_id: str = "zhiyingzou0202/gemma-3-4b-pt-object-detection" # "sergiopaniego/SmolVLM-256M-Instruct-object-detection"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 # Change to torch.bfloat16 for "google/gemma-3-4b-pt"

    batch_size: int = 1 # 8 for "google/gemma-3-4b-pt"
    learning_rate: float = 2e-05
    epochs = 1

