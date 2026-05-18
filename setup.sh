#!/bin/bash

# ==============================================================================
# Repo installation script
# ==============================================================================

log() { echo -e "\033[1;32m[INFO]\033[0m $1"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $1"; }
error() {
  echo -e "\033[1;31m[ERROR]\033[0m $1"
  exit 1
}

set -Eeuo pipefail

PYTORCH_ROCM_ARCH=gfx1100
WORKING_NAME="unifolm-wma-${REPO_BRANCH}"

log "Running branch-specific setup for: $REPO_BRANCH"

log "Setting up $WORKING_NAME conda env"
eval "$("${CONDA_PATH}/bin/conda" 'shell.bash' 'hook')"


if ! conda info --envs | grep -q ${WORKING_NAME}; then
  conda create --yes -n ${WORKING_NAME} python=3.10.18 -y
fi

conda activate ${WORKING_NAME}

log "Installing requirements"
conda install pinocchio=3.2.0 -c conda-forge -y
conda install ffmpeg=7.1.1 -c conda-forge

log "Installing Project"
cd ${MAIN_DIR}
git submodule update --init --recursive

pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.2
pip install -U xformers --index-url https://download.pytorch.org/whl/rocm7.2

log "Test xformers"
python -m xformers.info

pip install -e .

cd external/dlimp
pip install -e .

cd ../../..

echo "Branch-specific setup complete."
