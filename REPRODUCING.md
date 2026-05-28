# Reproducing mmore-medxpertqa-bench from 0 (Clariden)

This document walks through the full recipe to recreate the results in
`manifests/expected_results.yaml`. It assumes a CSCS account with access
to the `a127` Slurm account and the SwissAI scratch tree
(`/iopsstor/scratch/cscs/$USER`). The 7B run is ~5h on 1 GH200; the 72B
run is ~7h on 4 GH200s. Data fetching + indexing is ~15 min CPU + GPU.

## Prerequisites

- Slurm access to a partition with GH200 nodes (`normal`, `account=a127`).
- Python 3.11 (a venv will be created from `pyproject.toml`).
- ~15 GB free on `$SCRATCH` for HF model caches + MedPix images + question
  images + Qdrant ephemeral data.
- HuggingFace login if you haven't already accepted Qwen2.5-VL terms:
  `huggingface-cli login`.
- A built `qdrant` binary (we use 1.x compiled from source — see step 2).

## Step 1 — clone with submodule

```bash
cd /capstor/store/cscs/swissai/a127/homes/$USER
git clone --recurse-submodules <path-or-url> mmore-medxpertqa-bench
cd mmore-medxpertqa-bench
```

If you forgot `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

Verify the submodule HEAD matches what's pinned:

```bash
git submodule status
# expected: 8a66bdc... third_party/mmore-pr281 (remotes/origin/HEAD)
```

## Step 2 — build the Qdrant binary

```bash
git clone https://github.com/qdrant/qdrant.git qdrant-src
cd qdrant-src
git checkout v1.13.0       # any 1.x release works; v1.13 is what we used
cargo build --release      # ~5 min
cd ..
```

The eval sbatch expects the binary at
`$REPO_ROOT/qdrant-src/target/release/qdrant`; override by setting
`QDRANT_BIN` before sourcing `slurm/env.sh`.

## Step 3 — set up the Python venv

```bash
# Create a 3.11 venv (uv or python -m venv both fine)
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# Verify the pinned mmore source is importable from the submodule
python -c "
import sys; sys.path.insert(0, 'third_party/mmore-pr281/src')
import mmore
print('mmore from:', mmore.__file__)
from mmore.rag.model.vision import VLLMMultimodalAdapter
print('VLLMMultimodalAdapter:', VLLMMultimodalAdapter)
"
```

Source the shared env helper:

```bash
source slurm/env.sh
# prints REPO_ROOT, MMORE_SRC, HF_HOME, SCRATCH, QDRANT_BIN
```

## Step 4 — fetch + materialize data

### MedXpertQA-MM (the 2000-q test split + per-question images)

```bash
python data/load_medxpertqa_mm.py \
    --n 2000 --seed 42 \
    --images-dir $SCRATCH/medxpertqa_mm/images \
    --out  data/medxpertqa_mm_2000.jsonl
# ~5 min; ~3 GB of images materialized
```

Output: `data/medxpertqa_mm_2000.jsonl` (2000 rows, each with embedded
question text, options dict, gold label, and `image_paths` resolved
absolute under `$SCRATCH/medxpertqa_mm/images`).

### MedPix-2.0 (the retrieval corpus)

```bash
python data/load_medpix.py --out-dir $SCRATCH/medpix2
# ~10 min; ~700 MB of images materialized
```

Output:
- `$SCRATCH/medpix2/images/MPX*.png` — 2050 case images
- `$SCRATCH/medpix2/medpix_cases.parquet` — per-image rows with case metadata
- `$SCRATCH/medpix2/medpix_case_text.parquet` — per-case consolidated text
  chunks in mmore's `(pdf_path, page_number, text)` schema

## Step 5 — run the eval

The sbatch handles Qdrant startup, indexing, eval, and cleanup. Submit
the 7B first (faster, single GPU, exposes any setup issues cheaply):

```bash
VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
  VLM_TP=1 \
  QUESTIONS_FILE=$REPO_ROOT/data/medxpertqa_mm_2000.jsonl \
  MEDPIX_DIR=$SCRATCH/medpix2 \
  sbatch --gpus=1 slurm/eval.sbatch
```

Then the 72B:

```bash
VLM_MODEL=Qwen/Qwen2.5-VL-72B-Instruct \
  VLM_TP=4 \
  QUESTIONS_FILE=$REPO_ROOT/data/medxpertqa_mm_2000.jsonl \
  MEDPIX_DIR=$SCRATCH/medpix2 \
  sbatch --gpus=4 slurm/eval.sbatch
```

Watch progress:

```bash
JOBID=<from sbatch output>
tail -F logs/eval_${JOBID}.{out,err}
```

Outputs land under `results/eval_${JOBID}/`:
- `results.jsonl` — one record per (question, condition) with per-parser
  predictions, retrieval sources, body-system metadata
- `summary.json` — accuracy per condition + per-body-system + per-parser

## Step 6 — audit

### Paired McNemar within the run

```bash
python audit/mcnemar.py --results-jsonl results/eval_${JOBID}/results.jsonl
```

Prints accuracy + Wilson CI per condition; pairwise McNemar with Holm
correction; per-body-system breakdown; discordant-pair head-counts.

### Gold-coverage subset

```bash
python audit/gold_coverage.py \
    --results-jsonl   results/eval_${JOBID}/results.jsonl \
    --questions-jsonl data/medxpertqa_mm_2000.jsonl \
    --cases-parquet   $SCRATCH/medpix2/medpix_cases.parquet
```

Filters to questions whose correct-option text appears in MedPix
titles/case_diagnoses and re-scores on that subset (~203/2000 = 10.2%).

### Perceptual-hash contamination check (optional)

```bash
python audit/contamination_phash.py \
    --results-jsonl   results/eval_${JOBID}/results.jsonl \
    --questions-jsonl data/medxpertqa_mm_2000.jsonl \
    --cases-parquet   $SCRATCH/medpix2/medpix_cases.parquet \
    --condition       use_vision-on
```

Tests whether image-RAG correct-answer cases are near-duplicates of the
question image (which would indicate data leakage). Strict threshold
(aH ≤ 10 AND dH ≤ 10) should report 0 hits.

## Step 7 — compare to the manifest

```bash
cat results/eval_${JOBID}/summary.json
cat manifests/expected_results.yaml
```

A reproduction is considered "good" if:
- Each condition's accuracy lands within ±2pp of the manifest value
  (Wilson 95% CI at n=2000).
- Sign of every RAG-vs-no-RAG delta matches the manifest (i.e. all
  three conditions remain null or mildly-negative).
- The 72B no-RAG cell lands within ±2pp of Hu et al. 2025's 29.95%.

Individual p-values vary slightly from sampling noise even at temp=0
(vLLM scheduling differences); use them as orientation, not as identity
checks.

## Common issues

### vLLM can't find libcudart.so.13

The Cu13 lib ships inside the venv at
`.venv/lib/python3.11/site-packages/nvidia/cu13/lib`. `slurm/env.sh`
adds this to `LD_LIBRARY_PATH` automatically; if you run the eval
outside sbatch you need to source it manually.

### Qdrant fails to start

Usually a stale lock file under `$SCRATCH/qdrant_bench_<JOBID>`. The
sbatch cleans up on EXIT but a killed job leaves it behind:

```bash
rm -rf $SCRATCH/qdrant_bench_*
```

### `use_vision-on` shows many null predictions on 7B

Expected. PR #281's default `max_images_per_request=20` floods the 7B
with 5-10 retrieved images; the small model loses focus and exhausts
the 1024-token output budget without emitting `\boxed{}`. ~70/2000 nulls
on the 7B; ~0 on the 72B. This is a documented framework-default
characteristic, not a port bug — see the paper's Phase 8 caveat.

### Out of disk on `$HOME`

The HF cache symlink should put weights on `$SCRATCH`:

```bash
ls -la ~/.cache/huggingface
# should show: huggingface -> /iopsstor/scratch/cscs/$USER/hf_cache
```

If not:

```bash
mkdir -p $SCRATCH/hf_cache
mv ~/.cache/huggingface/* $SCRATCH/hf_cache/  2>/dev/null || true
rm -rf ~/.cache/huggingface
ln -s $SCRATCH/hf_cache ~/.cache/huggingface
```
