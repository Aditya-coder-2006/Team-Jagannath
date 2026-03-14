import cv2
import os
import numpy as np
from typing import List, Union
from pathlib import Path

class FrameExtractor:
    """
    Extracts frames from a video file or reads a folder of frames.
    Handles corrupted frames and very short videos.
    """
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def extract_from_video(self, video_path: str) -> List[np.ndarray]:
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)

        # 🔧 FIX: skip corrupted videos
        if not cap.isOpened():
            print(f"[WARNING] Skipping corrupted video: {video_path}")
            return [np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)]

        frames = []

        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.target_size)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            print(f"[WARNING] No frames extracted from: {video_path}")
            frames = [np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)]

        return frames

    def read_from_directory(self, dir_path: str) -> List[np.ndarray]:
        valid_extensions = {".jpg", ".jpeg", ".png"}
        files = sorted([f for f in os.listdir(dir_path) if Path(f).suffix.lower() in valid_extensions])
        
        frames = []
        for file in files:
            file_path = os.path.join(dir_path, file)
            frame = cv2.imread(file_path)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size)
                frames.append(frame)
        
        if len(frames) == 0:
            frames = [np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)]
            
        return frames

    def get_frames(self, input_path: str) -> List[np.ndarray]:
        try:
            if os.path.isdir(input_path):
                return self.read_from_directory(input_path)
            else:
                return self.extract_from_video(input_path)
        except Exception as e:
            print(f"[ERROR] Failed loading {input_path}: {e}")
            return [np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)]