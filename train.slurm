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

module load StdEnv/2023
module load python/3.10
module load cuda/11.7
module load scipy-stack
module load pytorch/2.2

VENV_DIR="$SLURM_TMPDIR/venv"
python -m venv $VENV_DIR
source $VENV_DIR/bin/activate

pip install --upgrade pip
pip install -r requirements.txt || exit 1

python main.py
