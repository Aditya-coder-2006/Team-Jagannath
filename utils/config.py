import os
import cv2
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    num_frames: int = 32
    feature_dim: int = 2048
    flow_dim: int = 2
    fused_dim: int = 1024
    transformer_heads: int = 8
    transformer_layers: int = 4
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 6
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    triplet_margin: float = 1.0
    consistency_weight: float = 0.5
    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    sampling_method: str = "uniform"
    max_train_samples: int = 500
    max_val_samples: int = 100

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data_dir: str = "C:\\Users\\adity\\Downloads\\ml-ware-26-sherlock-files"
    output_dir: str = "checkpoints"
    use_wandb: bool = True
    wandb_project: str = "video-chronological-sorting"

def load_config(config_path: str) -> Config:
    if not os.path.exists(config_path):
        return Config()
    
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
        
    config = Config()
    if "model" in data:
        config.model = ModelConfig(**data["model"])
    if "training" in data:
        config.training = TrainingConfig(**data["training"])
    if "data_dir" in data:
        config.data_dir = data["data_dir"]
    if "output_dir" in data:
        config.output_dir = data["output_dir"]
    if "use_wandb" in data:
        config.use_wandb = data["use_wandb"]
        
    return config

def setup_directories(config: Config):
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "logs"), exist_ok=True)
