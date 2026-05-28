"""
Materialise MedPix-2.0 from HuggingFace to local disk + emit:

  1. {out_dir}/images/{u_id}_{image_idx}.png — one image per (u_id, idx)
  2. {out_dir}/medpix_cases.parquet         — one row per (u_id, image_idx) with
     all case metadata + image_path
  3. {out_dir}/medpix_case_text.parquet     — one row per unique u_id with
     a consolidated case-text chunk (title, history, findings, diagnosis,
     discussion, etc.) keyed under the column name `pdf_path = u_id` for
     drop-in compatibility with the existing mmore text-indexing schema

Source: architojha/medpix-2.0-dataset on the HF Hub
(NLM MedPix repackaged; free for research use).

671 unique cases × ~3 multi-view images each = 2050 images total
(~700 MB at PNG).

Usage:
    python data/load_medpix.py --out-dir $SCRATCH/medpix2
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("load_medpix")


DATASET_ID = "architojha/medpix-2.0-dataset"


def build_case_text(row: pd.Series) -> str:
    """Single text chunk per case for the text retriever. Concatenates
    the meaningful clinical fields with light section headers so the
    text retriever has the kind of structured prose its retriever was
    trained on."""
    parts: List[str] = []
    title = (row.get("title") or "").strip()
    case_dx = (row.get("case_diagnosis") or "").strip()
    ddx = (row.get("differential_diagnosis") or "").strip()
    if title:
        parts.append(f"Diagnosis: {title}")
    if case_dx and case_dx != title:
        parts.append(f"Case diagnosis: {case_dx}")
    if ddx and ddx != case_dx:
        parts.append(f"Differential: {ddx}")
    for label, field in [
        ("Modality", "modality"),
        ("Plane", "plane"),
        ("Location", "location"),
    ]:
        v = (row.get(field) or "").strip()
        if v:
            parts.append(f"{label}: {v}")
    age = (row.get("age") or "").strip()
    sex = (row.get("sex") or "").strip()
    if age or sex:
        parts.append(f"Patient: {age or '?'}yo {sex or '?'}")
    for label, field in [
        ("History", "history"),
        ("Exam", "exam"),
        ("Findings", "findings"),
        ("Discussion", "disease_discussion"),
    ]:
        v = (row.get(field) or "").strip()
        if v and v != "N/A":
            parts.append(f"{label}: {v}")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True,
                    help="Destination directory (e.g. $SCRATCH/medpix2)")
    ap.add_argument("--dataset-id", default=DATASET_ID)
    ap.add_argument("--cache-dir", default=None,
                    help="HF datasets cache (default: HF_HOME)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s from HF...", args.dataset_id)
    from datasets import load_dataset

    ds = load_dataset(args.dataset_id, cache_dir=args.cache_dir)
    # The dataset has a single 'train' split with rows = case images.
    split = list(ds.keys())[0]
    log.info("Split=%s, %d rows", split, len(ds[split]))

    # MedPix groups multiple images per case (same u_id). We materialise
    # each image once on disk, indexed by (u_id, image_idx). image_idx is
    # the per-case sequence — preserve insertion order from the dataset.
    rows: List[Dict[str, Any]] = []
    per_case_counter: Dict[str, int] = defaultdict(int)
    for i, r in enumerate(ds[split]):
        uid = str(r.get("u_id"))
        if not uid:
            continue
        image_idx = per_case_counter[uid]
        per_case_counter[uid] += 1
        img = r.get("image")
        if img is None:
            continue
        path = images_dir / f"{uid}_{image_idx:02d}.png"
        if not path.exists():
            # img is a PIL.Image-like object from HF datasets
            img.convert("RGB").save(path, format="PNG")

        row: Dict[str, Any] = {k: r.get(k) for k in r.keys() if k != "image"}
        row["u_id"] = uid
        row["image_idx"] = image_idx
        row["image_path"] = str(path)
        rows.append(row)
        if (i + 1) % 200 == 0:
            log.info("  materialised %d/%d images", i + 1, len(ds[split]))

    df = pd.DataFrame(rows)
    n_unique = df["u_id"].nunique()
    log.info("Materialised %d images across %d unique cases", len(df), n_unique)

    # Count images per case → record uid_n_images for downstream filters
    counts = df.groupby("u_id").size().rename("uid_n_images").reset_index()
    df = df.merge(counts, on="u_id", how="left")

    cases_path = out / "medpix_cases.parquet"
    df.to_parquet(cases_path, index=False, compression="zstd")
    log.info("Wrote %s (%d rows, %d unique cases, %.2f MB)",
             cases_path, len(df), n_unique,
             cases_path.stat().st_size / 1e6)

    # Per-case consolidated text chunk: dedupe by u_id (each case appears
    # multiple times in df, one per image; the per-case clinical fields
    # are identical across image rows so taking the first row is safe).
    cases = df.drop_duplicates(subset="u_id", keep="first").copy()
    cases["text"] = cases.apply(build_case_text, axis=1)
    text_out = pd.DataFrame({
        # `pdf_path = u_id` keeps the existing text-indexer schema usable
        # without modification (it expects pdf_path / page_number / text).
        "pdf_path":    cases["u_id"].astype(str),
        "page_number": 0,
        "text":        cases["text"],
    })
    text_path = out / "medpix_case_text.parquet"
    text_out.to_parquet(text_path, index=False, compression="zstd")
    log.info("Wrote %s (%d cases, %.2f MB, median %d chars/case)",
             text_path, len(text_out),
             text_path.stat().st_size / 1e6,
             int(text_out["text"].str.len().median()))


if __name__ == "__main__":
    main()
