#!/bin/bash

# Define Python path
PYTHON="/cta/users/mert.coskuner/.conda/envs/openfgl/bin/python"

echo "Using Python: $PYTHON"

# Install OpenFGL in editable mode to ensure changes are picked up
echo "Installing OpenFGL in editable mode..."
$PYTHON -m pip install -e .

# Run Reproduction
echo "========================================================"
echo "Running FedALA Reproduction (Table 7 Baseline)"
echo "========================================================"
$PYTHON run_reproduction.py

# Run FedALA-SA
echo "========================================================"
echo "Running FedALA-SA (3 Options)"
echo "========================================================"
$PYTHON run_fedala_sa.py

echo "All experiments completed."
