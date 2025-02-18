#!/bin/bash
# scripts/run_train.sh
# This script launches the GRPO training process.

# Exit immediately if a command exits with a non-zero status.
set -e

# (Optional) Activate your virtual environment if you use one.
# Uncomment and adjust the following line if necessary.
# source /path/to/your/venv/bin/activate

# Optionally, set PYTHONPATH to include the src/ directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Print the current configuration (optional)
echo "Starting GRPO training..."
echo "Using Python: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"

# You can run the training script directly:
python src/train.py

# Alternatively, if you are using Hugging Face Accelerate, you can run:
# accelerate launch src/train.py

echo "Training script finished."
