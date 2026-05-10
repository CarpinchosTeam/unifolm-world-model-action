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

#[ -z ${FLASH_ATTENTION+x} ] && FLASH_ATTENTION=FALSE
#[ -z ${FLASH_ATTENTION_TRITON_AMD_ENABLE+x} ] && FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
PYTORCH_ROCM_ARCH=gfx1100

log "Running branch-specific setup for: $REPO_BRANCH"

log "Setting up unifolm-wma-${REPO_BRANCH} conda env"
eval "$("${CONDA_PATH}/bin/conda" 'shell.bash' 'hook')"

# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

if ! conda info --envs | grep -q "unifolm-wma-${REPO_BRANCH}"; then
  conda create --yes -n unifolm-wma-${REPO_BRANCH} python=3.10.19
fi

conda activate unifolm-wma-${REPO_BRANCH}

log "Installing requirements"
conda install pinocchio=3.2.0 -c conda-forge -y
conda install ffmpeg=7.1.1 -c conda-forge

log "Installing Project"
cd ${MAIN_DIR}
git submodule update --init --recursive

uv pip install -e .

cd external/dlimp
uv pip install -e .

cd ../../..

echo "Branch-specific setup complete."
