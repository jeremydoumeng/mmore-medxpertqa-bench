# Handoff — mmore multimodal RAG benchmark on Alps GH200

This document captures everything a fresh session needs to continue the work.
The benchmark goal is to add results to the mmore paper:
**mmore-multimodal (ColPali) vs mmore-text-only vs no-RAG** on
MedXpertQA Text questions, with a real medical PDF corpus, using
Meditron-70B via vLLM.

Read [README.md](README.md) for the Qdrant binary story and
[BENCHMARK.md](BENCHMARK.md) for the ColPali port history — both
have been kept up to date.

---

## Where we are right now (last action of the previous session)

- **End-to-end pipeline works on real medical corpus** (1000 PLOS PDFs,
  17,075 pages, indexed in both ColPali and text-only Qdrant collections).
- **Sanity at N=20 shows the case-continuation problem**: with
  retrieved PLOS context (which looks like medical case presentations),
  Meditron-70B tries to *continue with another case* instead of answering
  the multiple-choice question. no-RAG gives clean answers; RAG conditions
  give regurgitation/empty outputs in ~half the cases.
- **Just fixed**: prompt template now has hard `=== BEGIN/END REFERENCE
  MATERIAL ===` delimiters and an explicit "do NOT generate new questions"
  system instruction. Eval set scaled from N=20 to N=200 for statistical
  power.
- **Submitted**: `sbatch sanity_eval.sbatch` with these changes — but the
  last submit hit a tool-permission error, so the sbatch needs to be
  resubmitted at session start (next step below).

---

## Immediate next step for the new session

```bash
# Resubmit the sanity sbatch with the new prompt + 200 questions
sbatch $REPO_ROOT/sanity_eval.sbatch

# Watch
squeue -u $USER
tail -F $REPO_ROOT/logs/sanity_eval_<JOBID>.out
```

ETA ~30 min once it starts. The ColPali process step is cache-hit
(parquet already at `$SCRATCH/colpali_plos_1k_output/`).
Job will: launch qdrant → skip process (cache) → stream-index ColPali
(~6 min) → text-index (~7 min) → load Meditron-70B (~5 min) → generate
600 prompts (~15 min) → score.

Expected outcomes (with N=200):
- **If RAG > no-RAG**: prompt fix worked; scale to N=500 then write up.
- **If RAG ≈ no-RAG**: case-continuation reduced but not eliminated; try
  more prompt tweaks or report as a finding.
- **If RAG < no-RAG**: real concern; investigate retrieval relevance.

---

## Project layout

```
$HOME/
├── bench/                          # All benchmark work lives here
│   ├── HANDOFF.md                  # this file
│   ├── README.md                   # qdrant binary story
│   ├── BENCHMARK.md                # ColPali port story
│   ├── build_qdrant_alps.sh        # qdrant compilation recipe
│   ├── sanity_eval.sbatch          # main eval sbatch
│   ├── fetch_plos_corpus.sbatch    # corpus fetcher sbatch
│   ├── corpus/
│   │   ├── fetch_plos.py           # ← current working PDF fetcher
│   │   ├── fetch_pmc_oa.py         # ← BROKEN, NCBI deprecated FTP list
│   │   └── fetch_arxiv_qbio.py     # ← fallback, used for initial sanity
│   ├── eval/
│   │   ├── load_medxpertqa.py      # downloads MedXpertQA from HF
│   │   ├── medxpertqa_text_proto.jsonl   # 20-q sanity set
│   │   ├── medxpertqa_text_200.jsonl     # 200-q eval set (current target)
│   │   ├── run_eval.py             # 3-condition eval harness
│   │   ├── index_text_pages.py     # text-only indexer (page-level)
│   │   └── index_colpali_streaming.py    # ← streaming parquet reader
│   ├── configs/
│   │   ├── colpali_process_proto.yml
│   │   └── colpali_index_proto.yml
│   ├── qdrant-src/                 # source build of Qdrant v1.17.1
│   │   └── target/release/qdrant   # 72 MB binary, jemalloc LG_PAGE=16
│   ├── protoc/bin/protoc           # build-time dep
│   ├── mmore-qdrant/               # PR-branch clone (NOT actively used for eval)
│   ├── results/sanity_<JOBID>/     # per-job results.jsonl + summary.json
│   └── logs/                       # all sbatch logs
│
├── mmore-qdrant/                   # ← scratch tree — THE ONE THE EVAL USES
│   ├── .venv/                      # vllm 0.20 + cu126 + patched SPLADE
│   └── src/mmore/colpali/
│       ├── qdrantcolpali.py        # ← our new manager
│       ├── run_index.py            # patched: backend dispatch
│       ├── retriever.py            # patched: backend dispatch
│       └── milvuscolpali.py        # unmodified Milvus original
│
├── pr-dir/mmore-qdrant/            # CLEAN PR copy (don't touch)
│
└── $SCRATCH/  # NVMe scratch
    ├── medical_corpus_plos_1k/     # ← current corpus (1000 PLOS PDFs)
    │   └── manifest.json
    ├── medical_corpus_proto/       # arXiv sanity corpus (99 papers)
    ├── colpali_plos_1k_output/     # ← cached ColPali process output
    │   ├── pdf_page_objects.parquet  # 17,075 pages × N tokens × 128
    │   └── pdf_page_text.parquet
    ├── hf/                         # HF model cache (Meditron-70B here)
    └── pmc_cache/                  # leftover from failed NCBI attempt
```

**Critical**: the eval runs from the **scratch tree** (`mmore-qdrant/`), not
the bench clone (`bench/mmore-qdrant/`). The scratch venv has vllm 0.20
pre-installed; building it in the bench venv requires gcc-13 because the
system default gcc 7.5 is too old for vllm's C++ extensions.

---

## How to do the most common tasks

### Resubmit the sanity sbatch

```bash
sbatch $REPO_ROOT/sanity_eval.sbatch
```

### Inspect a finished job's results

```bash
RUN=2122671  # job id
ls $REPO_ROOT/results/sanity_$RUN/
cat $REPO_ROOT/results/sanity_$RUN/summary.json

# Look at raw Meditron outputs (debugging Meditron quirks)
python -c "
import json
for line in open('$REPO_ROOT/results/sanity_$RUN/results.jsonl'):
    r = json.loads(line)
    print(r['condition'], r['qid'], 'gold:', r['gold'], 'pred:', r['predicted'],
          '\n  raw:', r['raw_output'][:300])
"
```

### Edit the prompt and re-run

Prompt is in [eval/run_eval.py](eval/run_eval.py) — see `SYSTEM_PROMPT_*`
and `build_prompt()`. Just edit and resubmit; no rebuild needed.

### Scale corpus size

```bash
# Fetch more PLOS papers (1 min per ~50 papers)
sbatch $REPO_ROOT/fetch_plos_corpus.sbatch
# Defaults to N=1000. Override:
PLOS_N=5000 sbatch $REPO_ROOT/fetch_plos_corpus.sbatch
```

### Scale eval set size

```bash
# Generate a bigger sample
python \
  $REPO_ROOT/eval/load_medxpertqa.py \
  --n 500 --seed 42 \
  --out $REPO_ROOT/eval/medxpertqa_text_500.jsonl

# Then update sanity_eval.sbatch:
#   QUESTIONS=$BENCH/eval/medxpertqa_text_500.jsonl
```

---

## The 17 things that bit us — keep these in mind

### Environment / packaging

1. **NCBI/PMC is firewalled from login** but reachable from compute nodes.
   Anything that hits `ftp.ncbi.nlm.nih.gov` / `pmc.ncbi.nlm.nih.gov` must
   run via SLURM, not on the login node.

2. **NCBI deprecated their FTP bulk listing.** The old `oa_comm_use_file_list.csv`
   404s. PMC PDFs aren't directly accessible from anywhere — they're
   gated behind a JS "Preparing to download" page. The AWS S3 mirror
   (`pmc-oa-opendata`) only has txt + xml, no PDFs. **Use PLOS instead**
   (see `bench/corpus/fetch_plos.py`) — works fine.

3. **`python` is not on PATH** on these nodes; only `python3` and venv
   pythons. Always use the venv python explicitly:
   `python`.

4. **mmore's pyproject.toml excludes aarch64 from the CUDA torch index**
   (see `[tool.uv.sources]`). Installing `mmore[cu126]` on Alps yields
   `torch==X+cpu`. After any mmore install, force-reinstall torch from
   the PyTorch index:

   ```bash
   VIRTUAL_ENV=.venv uv pip install --force-reinstall \
       --index-url https://download.pytorch.org/whl/cu126 torch torchvision
   ```

5. **Don't try to install vllm in the bench venv.** vllm's C++ extensions
   need gcc-12+; system default is gcc 7.5. Use the **scratch venv**
   (`$REPO_ROOT/third_party/mmore-pr281/.venv`)
   which already has `vllm==0.20.0`.

6. **`pymilvus-model`'s SPLADE impl uses `tokenizer.batch_encode_plus()`**
   which was removed in transformers 5.x. Already patched in the scratch
   venv (one-line `sed` of
   `.venv/lib/python3.11/site-packages/pymilvus/model/sparse/splade_embedding/splade_impl.py`
   — replace `batch_encode_plus(` with `(`). If you ever recreate the
   venv, redo this.

### Qdrant server (the binary we compiled)

7. **The binary is Alps-specific.** Built with `JEMALLOC_SYS_WITH_LG_PAGE=16`
   to match the 64KB-page GH200 kernel. It crashes on 4KB-page systems
   (laptops, most cloud). Recipe: `bench/build_qdrant_alps.sh`.

8. **The server runs co-located with the workload inside each SLURM job**
   — not persistently. The sbatch starts qdrant in the background bound
   to `127.0.0.1:6333`, runs the pipeline, then kills it via `trap` on
   exit. **No inter-node communication; this is intentional.**

9. **gRPC default timeout is too short for ColPali queries on big indices**
   — 17k pages × 1000 token-vectors per page means MaxSim takes >5s.
   `QdrantColpaliManager` now passes `timeout=300` to QdrantClient. If
   you increase the corpus further, you may need to bump this again.

10. **gRPC, not REST, for inserts.** Qdrant's REST has a 32 MB JSON
    payload cap which trips on ColPali multi-vectors at any reasonable
    batch size. `QdrantColpaliManager` uses `prefer_grpc=True` and
    batch_size=4 by default.

11. **`vllm 0.20` needs `libcudart.so.13`** which is bundled in the venv
    at `.venv/lib/python3.11/site-packages/nvidia/cu13/lib/` but not on
    LD_LIBRARY_PATH by default. The sbatch sets `LD_LIBRARY_PATH` before
    activating the venv:

    ```bash
    VLLM_CU13_LIB=$MMORE/.venv/lib/python3.11/site-packages/nvidia/cu13/lib
    export LD_LIBRARY_PATH=$VLLM_CU13_LIB:${LD_LIBRARY_PATH:-}
    ```

12. **`vllm 0.20` auto-enables DeepGEMM (FP8 kernels) on GH200**, which
    fails because `deep_gemm` isn't installed. Meditron-70B is BF16; we
    don't need FP8. Disable in the sbatch:

    ```bash
    export VLLM_USE_DEEP_GEMM=0
    ```

### ColPali processing quirks

13. **bf16 ColPali produces NaN/Inf in ~11% of pages** (varies — sometimes
    248, sometimes 300 per 2353 pages). `QdrantColpaliManager.insert_from_dataframe`
    filters them out and logs the count. Not fatal; we lose ~11% of corpus.
    Could try float16 or float32 forward if this becomes a problem.

14. **pyarrow parquet List type overflows at ~2.1B floats** (32-bit
    offsets). With 17k pages × 1000 tokens × 128 floats = 2.2B floats —
    `pd.read_parquet` fails with `OSError: List index overflow`. Use
    `bench/eval/index_colpali_streaming.py` which uses
    `pyarrow.parquet.ParquetFile.iter_batches()` to stream the parquet.
    The sbatch already calls this, not `mmore colpali index`.

### Meditron / prompting quirks

15. **Meditron-70B is a continuation model**, not strongly chat-tuned.
    It tends to interpret retrieved medical-case content + a question as
    "more cases to generate" and emits new questions/cases instead of
    answering. Current prompt has hard `=== BEGIN/END ===` delimiters
    and explicit anti-continuation system instruction. May need further
    iteration.

16. **`max_tokens=16` is too short.** Meditron generates reasoning before
    committing to an answer. Use `--llm-max-tokens 512`.

17. **`temperature=0` triggers degenerate loops** like
    `### Explanation: ### Answer: ### Explanation: ### Answer: ...`.
    Use `temperature=0.1` (matches the user's existing `rag_query.py`).

### Other

- **GPU memory**: ColPali stays loaded in our Python process when we call
  vLLM. `run_eval.py` explicitly `del`s retrievers + calls
  `torch.cuda.empty_cache()` before `LLM(...)`. Also lowered
  `gpu_memory_utilization=0.75` for headroom.

- **Process collection_name vs config collection_name**: mmore's main
  `Retriever.retrieve()` needs `collection_name` passed per call; the
  config's `collection_name` is just the default. `run_eval.py` passes
  it explicitly.

- **Per-job qdrant data dir**: the sbatch uses
  `QDRANT__STORAGE__STORAGE_PATH=$SCRATCH/qdrant_proto_$SLURM_JOB_ID`
  so each job gets a fresh empty qdrant. Means we re-index per job.
  For repeated runs we could point at a persistent path; not done yet.

- **Login-node compute is forbidden.** All ML work must run via SLURM
  (sbatch or srun). The fetcher and config edits can run on login;
  anything that loads a model or touches a GPU cannot.

---

## What's been done

### Phase 0 — infrastructure (week 1)

- Built Qdrant server v1.17.1 from source with `JEMALLOC_SYS_WITH_LG_PAGE=16`
  to match Alps GH200's 64KB kernel pages. Documented in
  [build_qdrant_alps.sh](build_qdrant_alps.sh) + [README.md](README.md).
- Smoke-tested PR #283's server-mode Qdrant adapter on the binary.

### Phase 1 — port ColPali to Qdrant (week 2)

- Discovered mmore's ColPali path is wired directly to Milvus (separate
  code path from the main RAG adapter in PR #283), so it doesn't run on
  Alps at all.
- Wrote `QdrantColpaliManager` (~280 lines) using Qdrant's native
  multi-vector + MAX_SIM (server-side late interaction, no Python rerank).
  Backend dispatch wired into mmore's `colpali/run_index.py` and `colpali/retriever.py`.
- Synthetic smoke test ✓, real-PDF smoke test ✓ (32 pages, 33k vectors,
  correct topical retrieval).
- All documented in [BENCHMARK.md](BENCHMARK.md).

### Phase 2 — sanity check + scale-up (current)

- Designed Approach A: same Meditron-70B LLM across all three conditions
  (text-only input); the only thing that varies is *which pages get retrieved*.
- Built end-to-end sanity harness: corpus fetcher, eval-set loader, eval
  script, sbatch.
- Iterated through ~17 environment issues (see "things that bit us" above).
- First successful end-to-end sanity run: arXiv q-bio corpus, all three
  conditions ran, accuracy at noise (corpus mismatched to clinical questions).
- Pivoted to PLOS for real medical PDFs (NCBI deprecated, EuropePMC hangs,
  PLOS works).
- Fetched 1000 PLOS PDFs (PLOS Medicine + ONE + NTDs + Pathogens).
- Hit pyarrow 32-bit List overflow → wrote streaming indexer.
- Hit gRPC DEADLINE_EXCEEDED on ColPali query → bumped to 300s.
- Sanity on PLOS at N=20: pipeline runs, but RAG conditions hit the
  case-continuation problem (Meditron treats retrieved cases as templates
  to continue). no-RAG = 0/20, multimodal = 2/20, text-only = 1/20.
  All at noise level for 10-option MCQ.

### Just before handoff

- Rewrote prompt with hard `=== BEGIN/END ===` delimiters and
  "do NOT generate new questions" system instruction.
- Scaled eval set to N=200.
- Submission failed on tool-permission error; needs resubmit.

---

## Plan from here

### Now (immediately)

1. **Resubmit the sanity sbatch** with the new prompt + 200-question eval set
   (`sbatch sanity_eval.sbatch`).
2. **Read the raw outputs** afterwards to confirm the prompt fix reduces
   case-continuation:

   ```bash
   python -c "
   import json, collections
   r = [json.loads(l) for l in open('bench/results/sanity_<JOBID>/results.jsonl')]
   by_cond = collections.defaultdict(list)
   for x in r: by_cond[x['condition']].append(x)
   for c, xs in by_cond.items():
       n_correct = sum(1 for x in xs if x['correct'])
       n_empty = sum(1 for x in xs if not x['raw_output'].strip())
       n_question = sum(1 for x in xs if 'Question' in x['raw_output'] or
                       'A 20-year-old' in x['raw_output'] or
                       'A 30-year-old' in x['raw_output'])
       print(f'{c}: acc={n_correct/len(xs):.2f} empty={n_empty}/{len(xs)} regurg≈{n_question}/{len(xs)}')
   "
   ```

### Next (depending on N=200 result)

- **If RAG > no-RAG by >5%**: scale to N=500-1000 for paper-quality CIs.
  Run lives at `medxpertqa_text_500.jsonl` (don't yet exist; gen with
  `load_medxpertqa.py --n 500`).
- **If RAG ≈ no-RAG**: the corpus is at the right ballpark; need more
  papers (5000-10000). Fetch with
  `PLOS_N=5000 sbatch fetch_plos_corpus.sbatch`. Note: fetch is at 0.3 s/paper
  rate-limited, so 5000 papers = 25 min; 10000 = 50 min.
- **If RAG < no-RAG**: real concern. Investigate:
  - Retrieved chunks may be too long; truncate context more aggressively.
  - Different prompt structure (Q first, context as "background" appendix).
  - Look at retrieval relevance directly: for each question, score whether
    the retrieved page is even *about* the question's topic.

### After numbers stabilize

- Stratified breakdown by `body_system` and `medical_task` (already
  computed in `summary.json` — just present in writeup).
- Optional 4th condition: re-index PLOS as **QA pairs** (synthetic
  Q/A generated by LLM from each page) to reproduce the MIRIAD paper's
  RAG-MIRIAD vs RAG-Passage comparison on a different corpus.
- Write up: extend [BENCHMARK.md](BENCHMARK.md) with the experimental
  section, then port relevant figures/tables to the mmore paper.

---

## Open follow-ups that aren't blocking

- **The bf16 NaN/Inf rate (~11%) in ColPali**: not investigated. Could be
  fixed by switching the vision encoder forward to float16 or float32.
  Costs us 11% of corpus on every run.
- **Should the `QdrantColpaliManager` port be upstreamed?** PR #283 is
  pending review; this would be a natural follow-up. The code lives at
  `bench/mmore-qdrant/src/mmore/colpali/qdrantcolpali.py` (clean copy)
  and `~/mmore-qdrant/src/mmore/colpali/qdrantcolpali.py` (actually used).
  Decision deferred until after this benchmark concludes.
- **Persistent qdrant data dir**: currently per-job. Indexing 17k pages
  takes ~6 min so it's tolerable, but a persistent setup would save that
  per re-run.
- **The `pyproject.toml` aarch64 CUDA exclusion** in mmore could be filed
  upstream — it's almost certainly stale (PyTorch ships aarch64 CUDA
  wheels since 2024).

---

## Key job IDs from the previous session (for log archaeology)

| Job | What | Outcome |
|---|---|---|
| 2069417 | first end-to-end on arXiv (cu13 + DeepGEMM + OOM fixes) | first PASS, 0/20 noise |
| 2073581 | arXiv with prompt+parser v1 | 2/1/3 — found regurgitation |
| 2086176 | arXiv with prompt v3 + temp=0.1 | 1/2/0 — found degenerate loops |
| 2118846 | PLOS corpus fetch | 1000/1000 PDFs in 18 min |
| 2121447 | first PLOS sanity | hit pyarrow List overflow |
| 2122671 | PLOS with streaming + gRPC timeout | last completed: 2/1/0 noise on N=20 |
| _next_ | PLOS with new prompt + N=200 | TBD — this is what to resubmit |

Logs at `$REPO_ROOT/logs/sanity_eval_<JOBID>.{out,err}`.
Results at `$REPO_ROOT/results/sanity_<JOBID>/`.
