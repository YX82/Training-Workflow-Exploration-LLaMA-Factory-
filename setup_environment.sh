#!/bin/bash

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh

# Create environment
conda create -n llamafactory python=3.11 -y

# Activate environment
conda activate llamafactory

# Install LLaMA-Factory with dependencies
pip install -e ".[torch,metrics]"

# Launch Web UI
llamafactory-cli webui
