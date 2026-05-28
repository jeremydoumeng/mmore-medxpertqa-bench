# mmore-medxpertqa-bench

Internal Clariden-only benchmark of [swiss-ai/mmore PR #281](https://github.com/swiss-ai/mmore/pull/281)
("images extracted from documents as inputs in the final LLM answering")
ported onto the Qdrant fork of mmore, evaluated on the **MedXpertQA-MM**
test split with **Qwen2.5-VL** at two scales (7B and 72B).

## What this repo measures

Multimodal RAG over a teaching-file image corpus (MedPix-2.0), three
conditions:

- **`no-rag`** — VLM receives only the question text + question image(s).
  Baseline.
- **`use_vision-off`** — PR #281 with `LLMConfig.use_vision = False`: text
  retrieval against MedPix case texts, only retrieved **text** reaches
  the VLM.
- **`use_vision-on`** — PR #281 with `LLMConfig.use_vision = True`: same
  text-driven retrieval, but the **images attached to the retrieved text
  chunks** are also passed to the VLM (the feature PR #281 introduces).

The eval uses the zero-shot CoT prompt + `\boxed{X}` extraction so the
no-RAG cell calibrates against Hu et al. 2025
([arxiv:2507.11200](https://arxiv.org/abs/2507.11200), Table 1) at 0.2995
on MedXpert with Qwen2.5-VL-72B.

## Headline (from `manifests/expected_results.yaml`)

| Model | Cond | Acc | Wilson 95% CI | Δ vs no-RAG | Holm p |
|---|---|---|---|---|---|
| **Qwen2.5-VL-72B** | no-rag         | **30.10%** | [28.1, 32.2] | — | — |
| Qwen2.5-VL-72B     | use_vision-off | 28.55%     | [26.6, 30.6] | −1.55pp | 0.325 |
| Qwen2.5-VL-72B     | use_vision-on  | 29.05%     | [27.1, 31.1] | −1.05pp | 0.588 |
| **Qwen2.5-VL-7B**  | no-rag         | **22.35%** | [20.6, 24.2] | — | — |
| Qwen2.5-VL-7B      | use_vision-off | 21.45%     | [19.7, 23.3] | −0.90pp | 0.831 |
| Qwen2.5-VL-7B      | use_vision-on  | 21.05%     | [19.3, 22.9] | −1.30pp | 0.728 |

Net result: **PR #281's `use_vision=True` is null on this benchmark with
this corpus, at both model sizes.** The 72B no-RAG cell reproduces the
public anchor (Hu et al. 2025) within ±0.30pp.

A gold-coverage subset audit (questions whose answer concept demonstrably
appears in the MedPix corpus, 10.2% = 203/2000 questions) shows
**text-context RAG conditions lift +5-7pp on covered questions** —
Holm-borderline at n=203, consistent across both model sizes. See
`audit/gold_coverage.py` and the manifest for details.

## Repo layout

```
mmore-medxpertqa-bench/
├── README.md                       — this file
├── REPRODUCING.md                  — step-by-step recipe from 0
├── pyproject.toml                  — pinned deps
├── .gitmodules                     — pins third_party/mmore-pr281 SHA
├── third_party/
│   └── mmore-pr281/                — PR #281 ported onto mmore-qdrant
│                                     (committed at the SHA recorded by the submodule)
├── data/
│   ├── load_medxpertqa_mm.py       — fetch + materialize MM split + images
│   ├── load_medpix.py              — fetch + materialize MedPix-2.0
│   └── README.md                   — dataset licenses + sizes
├── benchmark/
│   ├── index_medpix.py             — build mmore-rag Qdrant index over MedPix
│   ├── run_eval.py                 — eval entry point (3 conditions × n=2000)
│   └── parsers.py                  — official + boxed + strict + lenient parsers
├── audit/
│   ├── mcnemar.py                  — within-result-file paired McNemar + Holm
│   ├── gold_coverage.py            — re-score on the gold-coverage subset
│   └── contamination_phash.py      — perceptual-hash near-duplicate check
├── slurm/
│   ├── env.sh                      — shared env setup (paths, HF cache, vLLM)
│   └── eval.sbatch                 — full pipeline (qdrant + index + eval)
├── manifests/
│   └── expected_results.yaml       — canonical numbers + Wilson CIs + tolerances
└── results/                        — gitignored eval output (results.jsonl + summary.json)
```

## Quick start

Full step-by-step in [`REPRODUCING.md`](REPRODUCING.md). Short version (assumes
you're on Clariden with a working `mmore-qdrant` venv at `.venv/` and a
`qdrant` binary on `$REPO_ROOT/qdrant-src/target/release/qdrant`):

```bash
# clone with submodule
git clone --recurse-submodules <path-or-url> mmore-medxpertqa-bench
cd mmore-medxpertqa-bench
source slurm/env.sh

# 1. fetch data (one-time, ~15 min)
python data/load_medxpertqa_mm.py \
    --n 2000 --seed 42 \
    --images-dir $SCRATCH/medxpertqa_mm/images \
    --out  data/medxpertqa_mm_2000.jsonl

python data/load_medpix.py --out-dir $SCRATCH/medpix2

# 2. run eval (7B ~5h on 1 GPU, 72B ~7h on 4 GPUs)
VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct  VLM_TP=1  sbatch --gpus=1 slurm/eval.sbatch
VLM_MODEL=Qwen/Qwen2.5-VL-72B-Instruct VLM_TP=4  sbatch --gpus=4 slurm/eval.sbatch

# 3. audit
python audit/mcnemar.py       --results-jsonl results/eval_<JOBID>/results.jsonl
python audit/gold_coverage.py --results-jsonl results/eval_<JOBID>/results.jsonl \
                              --questions-jsonl data/medxpertqa_mm_2000.jsonl \
                              --cases-parquet  $SCRATCH/medpix2/medpix_cases.parquet

# 4. compare to manifest
cat results/eval_<JOBID>/summary.json   # vs manifests/expected_results.yaml
```

## What's pinned where (for reproducibility)

- **`third_party/mmore-pr281` submodule SHA**: pinned by `.gitmodules` to
  the commit that contains PR #281 ported onto the Qdrant fork +
  `VLLMMultimodalAdapter` (one additive subclass of `BaseMultimodalLLM`
  added so the bench fits inside Slurm's 12h walltime cap). Cloning with
  `--recurse-submodules` checks out the exact same code we ran.
- **`pyproject.toml`**: pins all Python deps to the versions used in the
  canonical runs (`torch==2.11.0`, `vllm==0.20.0`, `transformers==5.8.0`,
  `colpali-engine==0.3.15`, etc.).
- **`manifests/expected_results.yaml`**: records the exact accuracies,
  Wilson CIs, and Holm-corrected p-values produced by the canonical runs,
  plus a tolerance band (±2pp absolute, sign-agreement on deltas) for
  judging whether a reproduction matches.

## Scope

This repo ships **only the PR #281 path** (text-driven retrieval +
image-attachments through the mmore-rag pipeline). The cross-implementation
McNemar against an image-driven custom harness (Phase 7 in the source
log) is not reproducible from this repo; see the paper for that
comparison.

## License

Code: see `LICENSE` (inherits from `swiss-ai/mmore`).
Data:
- MedXpertQA: per-source license (CC BY-NC-SA, see HF dataset card).
- MedPix-2.0: NLM-MedPix derivative, free for research use (see HF dataset card).
- This repo redistributes neither dataset; both are fetched at runtime via HF.
