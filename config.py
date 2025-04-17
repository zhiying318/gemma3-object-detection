from dataclasses import dataclass
import torch

@dataclass
class Configuration:
    dataset_id: str = "ariG23498/license-detection-paligemma"

    model_id:str = "google/gemma-3-4b-pt"
    pg2_id:str = "google/paligemma2-3b-mix-448"

    device:str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    batch_size: int = 8
    learning_rate: float = 1e-5