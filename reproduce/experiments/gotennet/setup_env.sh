#!/usr/bin/env bash
# Create the isolated GotenNet environment used by run_gotennet.py.
#
# GotenNet needs the PyG C++/CUDA extension stack (torch_scatter / torch_sparse /
# torch_cluster) built against a specific torch build, which does not coexist
# with the main project's torch.  We therefore keep it in a separate venv and
# talk to the rest of the pipeline only through the split CSVs and the results
# JSON (see run_gotennet.py).
#
# Target hardware for this repo: single NVIDIA A10G (CUDA 12.x driver).
# Usage:  bash setup_env.sh
set -euo pipefail

cd "$(dirname "$0")"

# 1) Python 3.10 venv (GotenNet requires >=3.10).
uv venv --python 3.10 .venv_goten
source .venv_goten/bin/activate

# 2) torch 2.5.1 + cu124 (the combination GotenNet is tested against).
uv pip install "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu124

# 3) PyG extension wheels matching torch 2.5.0+cu124.
uv pip install torch_scatter torch_sparse torch_cluster torch_spline_conv pyg_lib \
    -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

# 4) GotenNet (core) + the runner's own dependencies.
#    NOTE: gotennet.utils imports omegaconf and pytorch_lightning at import time,
#    so both are required even though we use a plain-PyTorch training loop.
uv pip install gotennet torch_geometric rdkit scikit-learn omegaconf pytorch_lightning

echo "Done. Activate with: source $(pwd)/.venv_goten/bin/activate"
echo "Then run e.g.:      python run_gotennet.py --seed 42 --smoke-test --endpoints hof"
