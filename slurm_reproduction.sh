#!/bin/bash
#SBATCH --job-name=fedala_repro
#SBATCH --output=slurm_repro_%j.out
#SBATCH --error=slurm_repro_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=cuda
#SBATCH --qos=cuda

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Date: $(date)"

# Define Python path
PYTHON="/cta/users/mert.coskuner/.conda/envs/openfgl/bin/python"

# Navigate to project directory
cd /cta/users/mert.coskuner/OpenFGL

# Install in editable mode (just in case)
$PYTHON -m pip install -e .

echo "Starting FedALA Reproduction..."
$PYTHON run_reproduction.py

echo "Done."
