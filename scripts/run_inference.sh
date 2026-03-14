#!/bin/bash

# Check if video path is provided
if [ -z "$1" ]
then
    echo "Usage: ./scripts/run_inference.sh <path_to_video_or_frame_folder>"
    exit 1
fi

VIDEO_PATH=$1

# Run inference module
python -m inference.predict_sequence --video_path "$VIDEO_PATH"
