import os
import argparse
import random
import numpy as np
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import sys
import time

# Ensure `project_root` is on sys.path so local imports work when run from repo root.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.video_sorter import VideoChronologicalSorter
from training.losses import LossPipeline
from training.trainer import Trainer
from utils.config import load_config, setup_directories

# Placeholder for actual dataset
from data_processing.frame_extractor import FrameExtractor
from data_processing.optical_flow import OpticalFlowExtractor
from data_processing.frame_sampler import FrameSampler


class VideoDataset(Dataset):
    def __init__(self, data_dir: str, config, labels_path: str = None, videos=None, sampling_method: str = None):
        self.data_dir = data_dir
        self.config = config
        
        # Load labels if available
        self.labels = {}
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)
        
        allowed = set(videos) if videos is not None else None
        discovery_start = time.time()
        self.videos = self._discover_samples(data_dir, allowed)
        print(
            f"Discovered {len(self.videos)} samples from {data_dir} "
            f"in {time.time() - discovery_start:.2f}s"
        )

        # ❌ REMOVED DEBUG LIMIT
        # self.videos = self.videos[:16]
            
        self.extractor = FrameExtractor()
        sampler_method = sampling_method or getattr(self.config.training, "sampling_method", "motion_aware")
        self.sampler = FrameSampler(target_frames=self.config.model.num_frames, method=sampler_method)
        self.flow_extractor = OpticalFlowExtractor()

    @staticmethod
    def _discover_samples(data_dir: str, allowed=None):
        image_extensions = {".jpg", ".jpeg", ".png"}
        video_extensions = {".mp4", ".avi", ".mov"}

        discovered = []

        try:
            entries = sorted(os.scandir(data_dir), key=lambda entry: entry.name)
        except FileNotFoundError:
            return discovered

        for entry in entries:
            path = entry.path
            if allowed is not None and path not in allowed:
                continue

            if entry.is_file() and Path(entry.name).suffix.lower() in video_extensions:
                discovered.append(path)
                continue

            if not entry.is_dir():
                continue

            try:
                child_names = os.listdir(path)
            except OSError:
                continue

            if any(Path(file_name).suffix.lower() in image_extensions for file_name in child_names):
                discovered.append(path)

        if discovered or allowed is not None:
            return discovered

        # Fallback for unexpected nested layouts.
        for root, _, files in os.walk(data_dir):
            has_images = any(Path(file_name).suffix.lower() in image_extensions for file_name in files)
            has_videos = any(Path(file_name).suffix.lower() in video_extensions for file_name in files)
            if has_images or has_videos:
                discovered.append(root)

        return sorted(set(discovered))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        
        # 1. Extract frames
        frames = self.extractor.get_frames(video_path)
        
        # 2. Extract flow and sample to fixed length
        flow_mags = self.flow_extractor.sequence_flow_magnitudes(frames)
        sampled_indices = self.sampler.sample(frames, flow_mags)
        sampled_frames = [frames[i] for i in sampled_indices]
        sampled_flows = self.flow_extractor.compute_sequence_flow(sampled_frames)
        
        if len(sampled_flows) > 0:
            sampled_flows.append(sampled_flows[-1])
        else:
            sampled_flows = [np.zeros((224, 224, 2)) for _ in sampled_frames]
            
        # Tensors
        frames_tensor = torch.tensor(np.stack(sampled_frames)).permute(0, 3, 1, 2).float() / 255.0
        flows_tensor = torch.tensor(np.stack([f.mean(axis=(0, 1)) for f in sampled_flows])).float()

        num_frames = self.config.model.num_frames
        
        video_id = Path(video_path).stem
        
        ranks = []
        if video_id in self.labels:
            chronological_order = self.labels[video_id]

            index_to_rank = {frame_idx: rank for rank, frame_idx in enumerate(chronological_order)}
            
            for idx in sampled_indices:
                ranks.append(index_to_rank.get(idx, idx))
        else:
            ranks = sampled_indices
            
        ranks = torch.tensor(ranks)
        
        # Pair sampling
        pair_i = torch.randint(0, num_frames, (10,))
        pair_j = torch.randint(0, num_frames, (10,))
        
        pair_targets = (ranks[pair_i] < ranks[pair_j]).long()
        
        # Triplet sampling
        trip_a = torch.randint(0, num_frames, (5,))
        trip_b = torch.randint(0, num_frames, (5,))
        trip_c = torch.randint(0, num_frames, (5,))
        
        trip_targets = torch.zeros(5, dtype=torch.long)

        for k in range(5):
            ta, tb, tc = trip_a[k], trip_b[k], trip_c[k]
            ra, rb, rc = ranks[ta].item(), ranks[tb].item(), ranks[tc].item()

            arr = np.array([ra, rb, rc])
            order = np.argsort(arr)

            mapping = {
                (0,1,2):0,
                (0,2,1):1,
                (1,0,2):2,
                (1,2,0):3,
                (2,0,1):4,
                (2,1,0):5
            }

            trip_targets[k] = mapping[tuple(order.tolist())]
        
        return {
            "video_path": video_path,
            "frames": frames_tensor,
            "flows": flows_tensor,
            "pair_i": pair_i,
            "pair_j": pair_j,
            "pair_targets": pair_targets,
            "trip_a": trip_a,
            "trip_b": trip_b,
            "trip_c": trip_c,
            "trip_targets": trip_targets,

            # ✅ FIXED TRUE ORDER
            "true_order": ranks.long()
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--data_dir", type=str, default="C:\\Users\\adity\\Downloads\\ml-ware-26-sherlock-files")
    parser.add_argument("--labels", type=str, default="C:\\Users\\adity\\Downloads\\ml-ware-26-sherlock-files\\train_labels.json")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--sampling_method", type=str, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_directories(config)
    
    dataset_base = os.environ.get("DATASET_PATH", args.data_dir)

    print(f"Loading dataset from: {dataset_base}")
    
    train_dir = os.path.join(dataset_base, "train")

    if not os.path.exists(train_dir):
        print(f"Warning: 'train' directory not found inside {dataset_base}. Using base directory.")
        train_dir = dataset_base
    
    labels_path = args.labels
    if not os.path.exists(labels_path):
        candidate_labels = [
            os.path.join(dataset_base, "train_labels.json"),
            os.path.join(train_dir, "train_labels.json"),
            os.path.join(dataset_base, "labels.json"),
            os.path.join(train_dir, "labels.json"),
        ]
        labels_path = next((p for p in candidate_labels if os.path.exists(p)), None)
        if labels_path:
            print(f"Using labels file: {labels_path}")
        else:
            print("Warning: labels file not found. Training will use sampled frame indices as pseudo-order labels.")

    # Load full dataset
    sampling_method = args.sampling_method or getattr(config.training, "sampling_method", "motion_aware")
    print("Scanning dataset entries...")
    full_dataset = VideoDataset(train_dir, config, labels_path=labels_path, sampling_method=sampling_method)

    if len(full_dataset) == 0:
        print(f"CRITICAL: No videos or image-frame folders found in {train_dir}")
        return

    # Train / Validation Split
    videos = list(full_dataset.videos)
    random.Random(42).shuffle(videos)
    split_idx = int(0.8 * len(videos))
    train_videos, val_videos = videos[:split_idx], videos[split_idx:]

    max_train_samples = args.max_train_samples
    if max_train_samples is None:
        max_train_samples = getattr(config.training, "max_train_samples", 0)
    if max_train_samples and max_train_samples > 0:
        train_videos = train_videos[:max_train_samples]

    max_val_samples = args.max_val_samples
    if max_val_samples is None:
        max_val_samples = getattr(config.training, "max_val_samples", 0)
    if max_val_samples and max_val_samples > 0:
        val_videos = val_videos[:max_val_samples]

    print(
        f"Selected subset sizes | train={len(train_videos)} | val={len(val_videos)} "
        f"| total_discovered={len(videos)}"
    )

    if len(val_videos) == 0 and len(train_videos) > 1:
        val_videos = [train_videos.pop()]
    elif len(train_videos) == 0 and len(val_videos) > 0:
        train_videos = [val_videos[0]]

    train_dataset = VideoDataset(
        train_dir,
        config,
        labels_path=labels_path,
        videos=train_videos,
        sampling_method=sampling_method
    )
    val_dataset = VideoDataset(
        train_dir,
        config,
        labels_path=labels_path,
        videos=val_videos,
        sampling_method=sampling_method
    )

    print(f"Train videos: {len(train_dataset)}")
    print(f"Validation videos: {len(val_dataset)}")

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = getattr(config.training, "num_workers", 0)
    pin_memory = bool(getattr(config.training, "pin_memory", False))
    persistent_workers = bool(getattr(config.training, "persistent_workers", False)) and num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    model = VideoChronologicalSorter(config)

    criterion = LossPipeline(
        consistency_weight=config.training.consistency_weight
    )
    
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion
    )
    
    trainer.run()


if __name__ == "__main__":
    main()
