# Video Chronological Sorting System

This repository provides a production-ready, modular PyTorch training and inference pipeline for predicting the correct chronological order of shuffled video frames. 

It is designed for high accuracy using a multi-modal feature fusion approach (ResNet50 + Optical Flow) coupled with a Temporal Transformer, Pairwise Comparator, and Triplet Module to maximize Kendall Tau ranking consistency.

## Architecture

The system pipeline is as follows:
1. **Frame Extraction**: OpenCV-based frame reading for both video files and frame folders.
2. **Smart Sampling**: Uniform or Motion-Aware sampling.
3. **CNN Features**: Pretrained ResNet50 backbone (2048-dim).
4. **Optical Flow**: Farneback dense optical flow computing motion magnitudes.
5. **Feature Fusion**: Deep fusion layer combining CNN semantics with Flow kinetics.
6. **Temporal Transformer**: Global attention across fused frame embeddings.
7. **Triplet Module**: Local 3-frame classification for permutation invariant reasoning.
8. **Pairwise Comparator**: Predicts binary conditional probability `P(frame_i < frame_j)`.
9. **Ranking Aggregation**: Accumulates prediction scores to sort frames.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Setup

Prepare your datasets in the following structure:
```
dataset/
    train/
        video_001.mp4
        video_002/
            frame_0.jpg
            ...
    test/
```

## Training

The system defaults to training for **50 epochs**. You can adjust this inside `configs/default_config.yaml`.

Before training, set your Weights & Biases API key. The key provided is `003e9d2346e6e6f525d0efa5264f34b273be3b01`.
On Windows PowerShell:
```bash
$env:WANDB_API_KEY="003e9d2346e6e6f525d0efa5264f34b273be3b01"
```
On Linux/macOS:
```bash
export WANDB_API_KEY="003e9d2346e6e6f525d0efa5264f34b273be3b01"
```

To run training:
```bash
bash scripts/run_training.sh
```
OR manually pointing to your dataset folder:
```bash
python -m training.train --data_dir "C:\Users\adity\Downloads\ml-ware-26-sherlock-files" --config configs/default_config.yaml
```

Training integrates directly with Weights & Biases for loss, accuracy, and Kendall tau tracking.

## Inference

Returns inference strings formatted specifically as: `video_id,"frame_order_sequence"`
```bash
bash scripts/run_inference.sh path/to/video.mp4
```

Example output:
`video,"0 1 2 5 4 3"`

## Generation and Validation

For hackathon/competition scoring, two scripts have been provided to manage your final predictions list in bulk:

**Generating Predictions CSV:**
```bash
python scripts/submission_generator.py --test_dir "C:\Users\adity\Downloads\ml-ware-26-sherlock-files\test" --output "submission.csv"
```
This iterates over all videos and outputs the correctly formatted CSV headers.

**Validating CSV formatting strictly:**
```bash
python scripts/validate_submission.py --csv "submission.csv"
```
Ensure you run this before submitting to catch any missing indices, non-integers, or duplicating errors instantly.

### API Usage
The prediction algorithm is completely modularized for frontend/React integration:

```python
from inference.predict_sequence import VideoPredictor

predictor = VideoPredictor(checkpoint_path="checkpoints/best_model.pth", config_path="configs/default_config.yaml")
order_string = predictor.predict("path/to/video/or/folder")
print(order_string) # "0 5 1 2 4 3"
```
