# Data

This directory holds the **fetchers**; the actual data lives on scratch
(too big to ship with the repo). After running `load_medxpertqa_mm.py`
and `load_medpix.py` you'll have:

| location | content | size |
|---|---|---|
| `data/medxpertqa_mm_2000.jsonl` | 2000-q MM split (text only) | ~10 MB |
| `$SCRATCH/medxpertqa_mm/images/` | per-question images materialised from the HF dataset | ~3 GB |
| `$SCRATCH/medpix2/images/` | 2050 MedPix case images | ~700 MB |
| `$SCRATCH/medpix2/medpix_cases.parquet` | per-image rows (u_id, image_idx, image_path, case metadata) | ~30 MB |
| `$SCRATCH/medpix2/medpix_case_text.parquet` | per-case consolidated text chunks (pdf_path=u_id, page_number=0, text) | ~5 MB |

## Datasets used

### MedXpertQA-MM

- Source: [`TsinghuaC3I/MedXpertQA`](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA), `MM` config.
- Paper: Zuo et al. 2025 ("MedXpertQA: A challenging multimodal medical reasoning benchmark").
- Test split: 2000 multi-image MCQ questions with embedded images
  (X-ray, ECG, histology, dermatology, gross anatomy, fundoscopy).
  Each question has 4-10 options; this repo's eval uses the
  10-option-aware parser (A-J).
- License: see the HF dataset card. Generally permissive for research use.

### MedPix-2.0

- Source: [`architojha/medpix-2.0-dataset`](https://huggingface.co/datasets/architojha/medpix-2.0-dataset).
- The original MedPix is an NLM (US National Library of Medicine) public
  teaching-file collection (free for research use). The HF dataset is a
  repackaging.
- 671 unique cases, ~3 multi-view images per case (2050 total), each with
  per-case clinical metadata (title, history, exam, findings,
  case_diagnosis, differential_diagnosis, disease_discussion, etc.).

## Reproducing the gold-coverage subset

The audit script `audit/gold_coverage.py` builds a binary `covered: bool`
per qid by matching the correct-option text against any MedPix case's
`title` or `case_diagnosis` field (strict substring + fuzzy token-set
ratio ≥ 90, with non-matchable golds filtered). It re-uses
`medxpertqa_mm_2000.jsonl` and `medpix_cases.parquet`; no separate data
is needed.

## Determinism

- The 2000-q jsonl is deterministic given `--seed 42 --n 2000` (samples
  the full test split in seeded random order).
- The MedPix `image_idx` is the dataset's insertion order; the HF
  dataset enumerates rows deterministically, so the same `(u_id,
  image_idx) → image_path` mapping reconstructs across runs.
- vLLM generation at `temperature=0` is *almost* deterministic — small
  scheduling differences across GPU counts can shift a handful of
  per-question predictions (we observed ~5-9 questions of net difference
  on no-RAG cells between independent runs of the same configuration).
  This is within the tolerance band documented in
  `manifests/expected_results.yaml`.
