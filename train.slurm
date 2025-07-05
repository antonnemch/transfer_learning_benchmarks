#!/bin/bash
#SBATCH --job-name=cnn_gridsearch
#SBATCH --account=def-emohamme
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=nemc5377@mylaurier.ca
#SBATCH --mail-type=ALL

module load StdEnv/2023
module load python/3.11
module load scipy-stack

VENV_DIR="$SLURM_TMPDIR/venv"
python -m venv $VENV_DIR
source $VENV_DIR/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt || exit 1

# Ensure Python can find local modules
export PYTHONPATH=$PYTHONPATH:$(pwd)

python -u main.py
