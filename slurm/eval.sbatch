#!/bin/bash
#SBATCH --job-name=mmore-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --account=a127
#
# mmore-medxpertqa-bench — full PR-#281 RAG eval over MedXpertQA-MM.
#
# Pipeline:
#   1. start Qdrant (ephemeral, one collection per job)
#   2. index MedPix-2.0 cases as mmore documents (text + image attachments)
#      via the ported PR-#281 Indexer
#   3. run eval over 3 conditions (no-rag, use_vision-off, use_vision-on)
#      using PR's Retriever + vision module + the vLLM-backed
#      BaseMultimodalLLM subclass shipped in the same submodule
#
# Required env (set BEFORE sbatch, or via --export):
#   VLM_MODEL              HF model id (default Qwen/Qwen2.5-VL-7B-Instruct)
#   VLM_TP                 tensor parallel (1 for 7B, 4 for 72B)
#   CONDITIONS             space-separated subset of "no-rag use_vision-off use_vision-on"
#   VLM_MAX_TOKENS         output budget (default 1024 — needs room for CoT)
#   QUESTIONS_FILE         path to medxpertqa_mm_2000.jsonl
#   MEDPIX_DIR             path to medpix2/ (containing medpix_cases.parquet + medpix_case_text.parquet)
#
# Submit example (7B):
#   VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct VLM_TP=1 \
#     QUESTIONS_FILE=$REPO_ROOT/data/medxpertqa_mm_2000.jsonl \
#     MEDPIX_DIR=$SCRATCH/medpix2 \
#     sbatch --gpus=1 slurm/eval.sbatch
#
# Submit example (72B):
#   VLM_MODEL=Qwen/Qwen2.5-VL-72B-Instruct VLM_TP=4 \
#     QUESTIONS_FILE=$REPO_ROOT/data/medxpertqa_mm_2000.jsonl \
#     MEDPIX_DIR=$SCRATCH/medpix2 \
#     sbatch --gpus=4 slurm/eval.sbatch

set -euo pipefail

# Resolve the repo root + source env helpers
SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SBATCH_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Activate the venv (must be set up before submission — see REPRODUCING.md)
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
        source "$REPO_ROOT/.venv/bin/activate"
    else
        echo "ERROR: no .venv at $REPO_ROOT/.venv. See REPRODUCING.md step 'Install Python deps'." >&2
        exit 1
    fi
fi

# shellcheck source=env.sh
source "$REPO_ROOT/slurm/env.sh"

VLM_MODEL=${VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
VLM_TP=${VLM_TP:-1}
CONDITIONS=${CONDITIONS:-"no-rag use_vision-off use_vision-on"}
VLM_MAX_TOKENS=${VLM_MAX_TOKENS:-1024}
QUESTIONS_FILE=${QUESTIONS_FILE:-$REPO_ROOT/data/medxpertqa_mm_2000.jsonl}
MEDPIX_DIR=${MEDPIX_DIR:-$SCRATCH/medpix2}

MEDPIX_CASES_PARQUET="$MEDPIX_DIR/medpix_cases.parquet"
MEDPIX_TEXT_PARQUET="$MEDPIX_DIR/medpix_case_text.parquet"

RESULTS=$REPO_ROOT/results/eval_${SLURM_JOB_ID:-local}
QDRANT_DATA=$SCRATCH/qdrant_bench_${SLURM_JOB_ID:-local}
COLLECTION=medpix_pr281
mkdir -p logs "$RESULTS"

echo "==== mmore-medxpertqa-bench eval ===="
echo "Job:           ${SLURM_JOB_ID:-local}  ($(date))"
echo "Node:          ${SLURMD_NODENAME:-$(hostname)}"
echo "Repo:          $REPO_ROOT"
echo "VLM:           $VLM_MODEL (tp=$VLM_TP)"
echo "Conditions:    $CONDITIONS"
echo "Questions:     $QUESTIONS_FILE ($(wc -l < $QUESTIONS_FILE) lines)"
echo "MedPix dir:    $MEDPIX_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

[ -f "$QUESTIONS_FILE" ]         || { echo "ERROR: $QUESTIONS_FILE missing — run data/load_medxpertqa_mm.py" >&2; exit 1; }
[ -f "$MEDPIX_CASES_PARQUET" ]   || { echo "ERROR: $MEDPIX_CASES_PARQUET missing — run data/load_medpix.py"  >&2; exit 1; }
[ -f "$MEDPIX_TEXT_PARQUET" ]    || { echo "ERROR: $MEDPIX_TEXT_PARQUET missing — run data/load_medpix.py"   >&2; exit 1; }
[ -x "$QDRANT_BIN" ]             || { echo "ERROR: $QDRANT_BIN not found or not executable"                 >&2; exit 1; }

python -c "
import torch, mmore
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}')
print(f'mmore from: {mmore.__file__}')
from mmore.rag.model.vision import VLLMMultimodalAdapter, aggregate_image_paths, load_images_from_paths
print('PR #281 vision module: OK')
"

# ---- Qdrant lifecycle ----------------------------------------------------
QDRANT_PID=""
cleanup() {
    if [ -n "$QDRANT_PID" ]; then
        kill "$QDRANT_PID" 2>/dev/null || true
        wait "$QDRANT_PID" 2>/dev/null || true
    fi
    rm -rf "$QDRANT_DATA" 2>/dev/null || true
}
trap cleanup EXIT

rm -rf "$QDRANT_DATA" && mkdir -p "$QDRANT_DATA"
echo "==== Starting qdrant server ===="
QDRANT__SERVICE__HOST=127.0.0.1 \
QDRANT__SERVICE__MAX_WORKERS=4 \
QDRANT__STORAGE__STORAGE_PATH="$QDRANT_DATA" \
    "$QDRANT_BIN" > logs/qdrant_${SLURM_JOB_ID:-local}.log 2>&1 &
QDRANT_PID=$!
for i in {1..15}; do
    sleep 1
    if curl -fsS http://127.0.0.1:6333/healthz > /dev/null 2>&1; then
        echo "qdrant ready after ${i}s"; break
    fi
done
curl -fsS http://127.0.0.1:6333/healthz > /dev/null \
    || { echo "qdrant did not become healthy" >&2; exit 1; }

# ---- Index MedPix -----------------------------------------------------
echo "==== Indexing MedPix via ported Indexer ===="
python "$REPO_ROOT/benchmark/index_medpix.py" \
    --cases-parquet "$MEDPIX_CASES_PARQUET" \
    --text-parquet  "$MEDPIX_TEXT_PARQUET" \
    --qdrant-url    http://127.0.0.1:6333 \
    --collection    "$COLLECTION" \
    --batch-size    64

# ---- Run eval ---------------------------------------------------------
echo "==== run_eval.py ===="
python "$REPO_ROOT/benchmark/run_eval.py" \
    --questions "$QUESTIONS_FILE" \
    --conditions $CONDITIONS \
    --qdrant-url http://127.0.0.1:6333 \
    --collection "$COLLECTION" \
    --top-k 3 \
    --vlm-model "$VLM_MODEL" \
    --vlm-tp $VLM_TP \
    --vlm-max-len 32768 \
    --vlm-max-tokens $VLM_MAX_TOKENS \
    --max-images-per-prompt 12 \
    --max-retrieved-images 20 \
    --results-jsonl "$RESULTS/results.jsonl" \
    --summary-json  "$RESULTS/summary.json"

echo "==== Summary ===="
python -c "
import json
s = json.load(open('$RESULTS/summary.json'))
print(f\"n={s['n_questions']}  vlm={s['vlm_model']}  framework={s['framework']}\")
print(f\"{'cond':18s}  acc(primary)  acc(boxed)  acc(official)  null(boxed)\")
for c, st in s['by_condition'].items():
    bp = st['by_parser']
    print(f\"  {c:16s}  {st['accuracy']:.4f}        {bp['boxed']['accuracy']:.4f}      {bp['official']['accuracy']:.4f}         {bp['boxed']['null_pred']}\")
"

echo "==== Job finished: $(date) ===="
