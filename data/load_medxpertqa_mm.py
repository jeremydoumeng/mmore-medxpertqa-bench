"""
Load MedXpertQA-MM from HuggingFace, write to JSONL for the eval harness.

The full 2000-question MM test split is the canonical run for this repo
(`--n 2000`). Smaller n (e.g. `--n 200`) is for quick smoke tests; it's
a strict random subset of the 2000-q set when seeds match.

Schema of each output row:
    {
        "id":           str,    # e.g. "MM-1234"
        "question":     str,    # full question text including formatted options
        "options":      dict,   # {"A": "...", "B": "...", ...}
        "label":        str,    # gold answer letter, e.g. "C"
        "medical_task": str,    # "Diagnosis" / "Treatment" / "Basic Medicine"
        "body_system":  str,
        "question_type":str,    # "Reasoning" / "Understanding"
        "images":       list,   # basenames as in the HF dataset's images.zip
        "image_paths":  list,   # absolute paths under --images-dir
    }

Embedded image bytes in the HF dataset are materialised to disk under
--images-dir on first run (~3 GB for the full 2000-q split).

Usage (canonical):
    python data/load_medxpertqa_mm.py \\
        --n 2000 --seed 42 \\
        --images-dir $SCRATCH/medxpertqa_mm/images \\
        --out  data/medxpertqa_mm_2000.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("load_medxpertqa")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=["Text", "MM"], default="MM",
                    help="Which MedXpertQA subset to load (this repo is MM-only)")
    ap.add_argument("--n", type=int, default=2000,
                    help="Number of questions to sample. 2000 = full test split.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument(
        "--full",
        action="store_true",
        help="Skip sampling — write the entire test split",
    )
    ap.add_argument(
        "--images-dir",
        default=None,
        help="(MM only) Local directory with the extracted images.zip — used to "
             "resolve each `images` basename to an absolute path. If omitted, "
             "only the basenames are written.",
    )
    args = ap.parse_args()

    # Imported here so the script can be imported without needing datasets installed.
    from datasets import load_dataset

    log.info("Loading MedXpertQA %s test split from HuggingFace...", args.config)
    ds = load_dataset("TsinghuaC3I/MedXpertQA", args.config, split="test")
    log.info("Loaded %d rows", len(ds))

    if args.full:
        sample = list(ds)
    else:
        rng = random.Random(args.seed)
        idx = rng.sample(range(len(ds)), k=min(args.n, len(ds)))
        sample = [ds[i] for i in idx]
        log.info("Sampled %d rows (seed=%d)", len(sample), args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images_dir) if args.images_dir else None
    if args.config == "MM" and images_dir is not None and not images_dir.is_dir():
        raise SystemExit(f"--images-dir {images_dir} not a directory")

    missing_imgs = 0
    with out.open("w") as f:
        for row in sample:
            rec = {
                "id": row["id"],
                "question": row["question"],
                "options": row["options"],
                "label": row["label"],
                "medical_task": row.get("medical_task", ""),
                "body_system": row.get("body_system", ""),
                "question_type": row.get("question_type", ""),
            }
            if args.config == "MM":
                imgs = list(row.get("images") or [])
                rec["images"] = imgs
                if images_dir is not None:
                    paths = []
                    for name in imgs:
                        p = images_dir / name
                        if not p.exists():
                            missing_imgs += 1
                        paths.append(str(p))
                    rec["image_paths"] = paths
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if missing_imgs:
        log.warning("%d image basenames could not be resolved under %s", missing_imgs, images_dir)
    log.info("Wrote %d records to %s", len(sample), out)


if __name__ == "__main__":
    main()
