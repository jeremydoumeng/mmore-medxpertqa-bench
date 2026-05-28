#!/bin/bash
# Shared environment setup for all slurm scripts in this repo.
#
# Sourcing this file sets:
#   REPO_ROOT       — absolute path to the cloned mmore-medxpertqa-bench
#   MMORE_SRC       — path to the pinned PR-#281 mmore source tree (submodule)
#   HF_HOME         — HuggingFace cache (model weights + datasets)
#   HF_DATASETS_CACHE
#   SCRATCH         — scratch storage root (Clariden default $SCRATCH or
#                     $IOPSSTOR_SCRATCH; override by setting SCRATCH before sourcing)
#   QDRANT_BIN      — path to the qdrant server binary
#   LD_LIBRARY_PATH — adds the venv's nvidia/cu13/lib so vLLM 0.20 can find libcudart.so.13
#
# This script is Clariden-specific (assumes shared scratch + an mmore-qdrant
# venv with vllm 0.20 already installed); see docs/CLARIDEN.md for the
# concrete paths.

# Resolve repo root from this script's location
SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SLURM_DIR/.." && pwd)"
export MMORE_SRC="$REPO_ROOT/third_party/mmore-pr281/src"

# Scratch: prefer $SCRATCH if set; otherwise fall back to Clariden's iopsstor
if [ -z "${SCRATCH:-}" ]; then
    if [ -d "/iopsstor/scratch/cscs/$USER" ]; then
        export SCRATCH="/iopsstor/scratch/cscs/$USER"
    elif [ -d "/scratch/$USER" ]; then
        export SCRATCH="/scratch/$USER"
    else
        echo "WARNING: SCRATCH is unset and no default scratch dir found. Set SCRATCH manually." >&2
    fi
fi
mkdir -p "$SCRATCH" 2>/dev/null || true

# HF cache lives on scratch so we don't blow the home quota
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" 2>/dev/null || true

# Qdrant binary — built from source under $REPO_ROOT/qdrant-src or supplied
# externally by setting QDRANT_BIN before sourcing.
export QDRANT_BIN="${QDRANT_BIN:-$REPO_ROOT/qdrant-src/target/release/qdrant}"

# vLLM 0.20 ships libcudart.so.13 inside the venv. Surface it to the loader.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    _CU13_LIB="$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cu13/lib"
    if [ -d "$_CU13_LIB" ]; then
        export LD_LIBRARY_PATH="$_CU13_LIB:${LD_LIBRARY_PATH:-}"
    fi
fi

# vLLM auto-enables DeepGEMM (FP8 kernels) on GH200; the multimodal models
# we use are BF16 so disable it to avoid the FP8 build dep.
export VLLM_USE_DEEP_GEMM=0

# Make the pinned mmore source importable from any Python invocation
export PYTHONPATH="$MMORE_SRC${PYTHONPATH:+:$PYTHONPATH}"

echo "[env.sh] REPO_ROOT=$REPO_ROOT"
echo "[env.sh] MMORE_SRC=$MMORE_SRC"
echo "[env.sh] HF_HOME=$HF_HOME"
echo "[env.sh] SCRATCH=$SCRATCH"
echo "[env.sh] QDRANT_BIN=$QDRANT_BIN"
