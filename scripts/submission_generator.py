import os
import sys
import argparse
import csv
from pathlib import Path

# Ensure `project_root` is on sys.path so local imports work when run from repo root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tqdm import tqdm
from inference.predict_sequence import VideoPredictor

def generate_submission(video_dir: str, checkpoint_path: str, config_path: str, output_csv: str = "submission.csv", max_videos: int | None = None):
    '''
    Takes a directory containing test videos, runs inference to get their chronological frame order,
    and outputs a formatted submission.csv.
    If `max_videos` is provided, only the first N found videos will be processed (useful for Kaggle test sets).
    '''
    print(f"Loading Model from {checkpoint_path}...")
    predictor = VideoPredictor(checkpoint_path, config_path)
    
    # Support popular video formats
    video_files = []
    for ext in [".mp4", ".avi", ".mov"]:
        video_files.extend(list(Path(video_dir).rglob(f"*{ext}")))
        
    if not video_files:
        print(f"No video files found in {video_dir}.")
        return
        
    if max_videos is not None and len(video_files) > max_videos:
        print(f"Limiting to first {max_videos} of {len(video_files)} videos for submission.")
        video_files = sorted(video_files)[:max_videos]

    print(f"Found {len(video_files)} test videos. Generating predictions...")
    
    results = []
    
    for v_path in tqdm(video_files, desc="Processing videos"):
        video_id = v_path.stem
        # Predictor already returns 1-indexed orders
        order_str = predictor.predict(str(v_path))
        results.append((video_id, order_str))
        
    print(f"Writing {len(results)} predictions to {output_csv}...")
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["video_id", "order"])
        for video_id, order in results:
            writer.writerow([video_id, order])
            
    print("Submission generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission.csv for test videos")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test videos")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to best model weights")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output filename")
    parser.add_argument("--max_videos", type=int, default=None, help="Only process this many videos (e.g. 120 for Kaggle test)")
    
    args = parser.parse_args()
    
    generate_submission(
        video_dir=args.test_dir,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_csv=args.output,
        max_videos=args.max_videos
    )