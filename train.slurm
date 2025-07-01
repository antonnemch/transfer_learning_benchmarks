#!/bin/bash
#SBATCH --job-name=cnn_gridsearch
#SBATCH --account=def-emohamme
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=nemc5377@mylaurier.ca
#SBATCH --mail-type=ALL

# Load required modules
# Step 1: Load base Python and CUDA
module load python/3.10
module load cuda/11.7

# Step 2: Load PyTorch that works with the above CUDA
module load pytorch/2.0

# 2. Create virtual environment (if not already created)
VENV_DIR="$SLURM_TMPDIR/venv"
python -m venv $VENV_DIR
source $VENV_DIR/bin/activate


pip install --upgrade pip
pip install -r requirements.txt

# Run your training script
python main.py