#!/usr/bin/env bash
# Create the isolated MoLFormer environment used by run_molformer.py.
#
# MoLFormer (IBM, Ross et al. 2022) uses the IDIAP pytorch-fast-transformers
# linear-attention transformer with rotary embeddings. The BOOM protocol
# (see experiments/molformer/readme.md) additionally swaps the original
# apex.optimizers.FusedLAMB for torch_optimizer.Lamb so the environment does
# not require CUDA/Apex. We reuse that substitution here.
#
# Target hardware: a single machine with any of CPU, CUDA, or Apple MPS. No CUDA
# extensions are built -- the linear-attention pathway used by the pretrained
# checkpoint is pure PyTorch and works across those backends.
#
# Vendored upstream MoLFormer code lives in
#   experiments/molformer/src/molformer-main/notebooks/pretrained_molformer/
# and is NOT modified. The runner adds that directory to sys.path at runtime.
#
# Usage:  bash setup_env.sh
set -euo pipefail

cd "$(dirname "$0")"

# 1) Python 3.10 venv (fast_transformers is known to build on 3.10; 3.11+ works too).
uv venv --python 3.10 .venv_molformer
source .venv_molformer/bin/activate

# 2) PyTorch (default wheel; provides CPU plus CUDA or MPS acceleration where available).
uv pip install "torch==2.5.1" "torchvision==0.20.1"

# 3) Runner dependencies. pytorch-fast-transformers builds a small C++ extension
#    (CPU-only; does NOT need CUDA). The pinned 0.4.0 release is
#    what the vendored MolFormer code was written against.
#
#    NOTE: pytorch-fast-transformers requires torch at build time; installing it
#    AFTER torch avoids ImportError during its setup.py.
uv pip install \
    "pytorch-lightning>=2.0,<2.6" \
    "transformers>=4.30,<5" \
    "datasets>=2.14" \
    "torch-optimizer>=0.3.0" \
    "rdkit>=2023.9" \
    "pandas>=2.0" \
    "scikit-learn>=1.3" \
    "pyyaml>=6.0" \
    "regex>=2023.0" \
    "einops>=0.7"

# pytorch-fast-transformers has C++ extensions; needs --no-build-isolation
# because the build script imports torch at setup time. That flag also means
# setuptools must already be present in the target env, so we install it
# (plus wheel + ninja) BEFORE building.
uv pip install "setuptools>=68" "wheel" "ninja"
uv pip install --no-build-isolation "pytorch-fast-transformers==0.4.0"

echo
echo "Done. Activate with: source $(pwd)/.venv_molformer/bin/activate"
echo
echo "Next steps:"
echo "  1. Download the pretrained MoLFormer checkpoint from"
echo "        https://ibm.box.com/v/MoLFormer-data"
echo "     Extract 'Pretrained MoLFormer.zip'. From it copy the file"
echo "        Pretrained\\ MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt"
echo "     into:"
echo "        reproduce/experiments/data/molformer_ckpts/"
echo "  2. Smoke-test on one endpoint:"
echo "        python run_molformer.py --smoke-test --endpoints hof"
echo "  3. Full run:"
echo "        python run_molformer.py --seed 42"
