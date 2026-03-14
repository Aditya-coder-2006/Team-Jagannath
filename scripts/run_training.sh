#!/bin/bash

# Ensure output directory exists
mkdir -p checkpoints

# Run training module
python -m training.train --config configs/default_config.yaml
