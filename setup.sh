#!/bin/bash

# ==============================================================================
# Repo installation script
# ==============================================================================

set -Eeuo pipefail

echo "Running branch-specific setup for: $REPO_BRANCH"

eval "$("${CONDA_PATH}/bin/conda" 'shell.bash' 'hook')"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

if ! conda info --envs | grep -q "unifolm-wma"; then
    conda create --yes -n unifolm-wma-${REPO_BRANCH} python=3.10.19
fi

conda activate unifolm-wma-${REPO_BRANCH}

conda install pinocchio=3.2.0 -c conda-forge -y
conda install ffmpeg=7.1.1 -c conda-forge

pip install ninja
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.1

cd ${MAIN_DIR}
git submodule update --init --recursive

pip install -e .

cd external/dlimp
pip install -e .

cd ../../..

echo "Branch-specific setup complete."
