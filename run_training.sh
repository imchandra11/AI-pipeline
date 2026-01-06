#!/bin/bash
# Bash script to run training with proper PYTHONPATH
# Usage: ./run_training.sh --config YourProjectName/configs/your_project.yaml

# Set PYTHONPATH to current directory
export PYTHONPATH=.

# Run the training script with all arguments
python Regression/mainfittest.py "$@"

