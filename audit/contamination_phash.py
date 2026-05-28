"""
Perceptual-hash contamination audit for Phase 7.

For each MedXpertQA-MM-200 question, compare the question image(s) against the
top-K retrieved MedPix case images via aHash + dHash. Distances near zero
indicate near-duplicate retrieval — likely contamination (MedPix figures
re-published in the source materials behind MedXpertQA).

Two distance views:
  - aHash 64-bit: average-pixel threshold on an 8×8 thumbnail (perceptual
    similarity, robust to mild crop/contrast/JPEG noise).
  - dHash 64-bit: gradient-direction hash on a 9×8 thumbnail (sensitive to
    structural near-duplicates).

A pair is "likely-duplicate" if BOTH hash distances are <= 10 (≈84% bit
agreement), which empirically catches re-encodings and tight crops without
flagging visually-similar-but-different cases.

We stratify by:
  - image-RAG correct (n=61 in job 2318294) vs wrong (n=139)
  - all (Q image × top-3 retrieved cases per Q) pairs

If the lift is contamination, we expect the 'correct' set to have many
likely-duplicate hits and the 'wrong' set to have nearly none.

Usage:
    python audit_contamination_phash.py \\
        --results-jsonl bench/results/sanity_2318294/results.jsonl \\
        --cases-parquet $SCRATCH/medpix2/medpix_cases.parquet \\
        --top-k 3
"""

from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image


def _to_gray_thumb(img: Image.Image, size: Tuple[int, int]) -> List[int]:
    g = img.convert("L").resize(size, Image.LANCZOS)
    return list(g.getdata())


def ahash(img: Image.Image) -> int:
    pixels = _to_gray_thumb(img, (8, 8))
    avg = sum(pixels) / 64
    bits = 0
    for i, p in enumerate(pixels):
        if p >= avg:
            bits |= 1 << i
    return bits


def dhash(img: Image.Image) -> int:
    pixels = _to_gray_thumb(img, (9, 8))
    bits = 0
    idx = 0
    for r in range(8):
        for c in range(8):
            left = pixels[r * 9 + c]
            right = pixels[r * 9 + c + 1]
            if left > right:
                bits |= 1 << idx
            idx += 1
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-jsonl", required=True)
    ap.add_argument("--questions-jsonl", required=True,
                    help="MedXpertQA-MM jsonl (carries image_paths per qid)")
    ap.add_argument("--cases-parquet", required=True,
                    help="MedPix cases.parquet built by data/load_medpix.py")
    ap.add_argument("--condition", default="use_vision-on",
                    help="Which condition's retrieved images to audit. The "
                         "default targets PR #281's vision-on cell where "
                         "case images get attached to the prompt.")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Top-K retrieved cases to check per question")
    ap.add_argument("--ahash-thresh", type=int, default=10)
    ap.add_argument("--dhash-thresh", type=int, default=10)
    ap.add_argument("--max-pairs", type=int, default=10_000)
    args = ap.parse_args()

    # ---- Index MedPix case images by u_id → list of image_paths -----------
    cases = pd.read_parquet(args.cases_parquet)
    case_images: Dict[str, List[str]] = defaultdict(list)
    for _, row in cases.iterrows():
        case_images[str(row["u_id"])].append(row["image_path"])
    print(f"Loaded {len(case_images)} cases with {sum(len(v) for v in case_images.values())} total images")

    # ---- Question image_paths ARE in the questions jsonl, not results ----
    q_image_paths: Dict[str, List[str]] = {}
    for line in open(args.questions_jsonl):
        q = json.loads(line)
        q_image_paths[q["id"]] = q["image_paths"]
    print(f"Loaded image_paths for {len(q_image_paths)} questions")

    # ---- Load records for the targeted condition --------------------------
    by_qid: Dict[str, dict] = {}
    for line in open(args.results_jsonl):
        r = json.loads(line)
        if r["condition"] == args.condition:
            by_qid[r["qid"]] = r
    print(f"Loaded {len(by_qid)} '{args.condition}' records")

    # ---- Hash all involved images (question + retrieved) once ------------
    questions_to_check = list(by_qid.values())
    needed_paths = set()
    for r in questions_to_check:
        for p in q_image_paths.get(r["qid"], []):
            needed_paths.add(p)
        for src in r.get("sources", [])[: args.top_k]:
            uid = src.get("u_id")
            if uid in case_images:
                for ip in case_images[uid]:
                    needed_paths.add(ip)

    print(f"Hashing {len(needed_paths)} images...")
    img_ahash: Dict[str, int] = {}
    img_dhash: Dict[str, int] = {}
    for i, p in enumerate(sorted(needed_paths)):
        try:
            with Image.open(p) as im:
                im.load()
                img_ahash[p] = ahash(im)
                img_dhash[p] = dhash(im)
        except Exception as e:
            print(f"  skip {p}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  hashed {i+1}/{len(needed_paths)}")

    # ---- For each Q, compute min distance over (Q image × case images) ---
    # We aggregate per-Q the MINIMUM (ahash_dist, dhash_dist) across all
    # (q_image, retrieved_case_image) pairs — the most-similar one.
    per_q: List[dict] = []
    for r in questions_to_check:
        q_paths = q_image_paths.get(r["qid"], [])
        min_a, min_d = 64, 64
        best_case = None
        best_q_path = None
        best_case_img = None
        for q_path in q_paths:
            if q_path not in img_ahash:
                continue
            qa = img_ahash[q_path]
            qd = img_dhash[q_path]
            for src in r.get("sources", [])[: args.top_k]:
                uid = src.get("u_id")
                if uid not in case_images:
                    continue
                for ip in case_images[uid]:
                    if ip not in img_ahash:
                        continue
                    da = hamming(qa, img_ahash[ip])
                    dd = hamming(qd, img_dhash[ip])
                    if da + dd < min_a + min_d:
                        min_a = da
                        min_d = dd
                        best_case = uid
                        best_q_path = q_path
                        best_case_img = ip
        per_q.append({
            "qid": r["qid"],
            "correct": bool(r.get("correct_official")),
            "min_ahash": min_a,
            "min_dhash": min_d,
            "best_case": best_case,
            "best_q_path": best_q_path,
            "best_case_img": best_case_img,
            "gold": r["gold"],
            "body_system": r.get("body_system", ""),
        })

    # ---- Summarise --------------------------------------------------------
    def dist_table(records):
        a = [r["min_ahash"] for r in records]
        d = [r["min_dhash"] for r in records]
        n = len(records)
        if n == 0:
            return None
        def percentiles(xs):
            xs = sorted(xs)
            return xs[0], xs[n // 4], xs[n // 2], xs[(3 * n) // 4], xs[-1]
        a_pc = percentiles(a)
        d_pc = percentiles(d)
        # likely-duplicate flag
        likely = [
            r for r in records
            if r["min_ahash"] <= args.ahash_thresh and r["min_dhash"] <= args.dhash_thresh
        ]
        return {
            "n": n,
            "ahash_pcs": a_pc,
            "dhash_pcs": d_pc,
            "n_likely_dup": len(likely),
            "frac_likely_dup": len(likely) / n,
            "likely": likely,
        }

    correct = [r for r in per_q if r["correct"]]
    wrong = [r for r in per_q if not r["correct"]]

    print()
    print("=== Hash distance distribution: image-RAG CORRECT vs WRONG ===")
    print(f"Thresholds for 'likely-duplicate': ahash<={args.ahash_thresh}, dhash<={args.dhash_thresh}")
    for name, recs in (("correct", correct), ("wrong", wrong)):
        s = dist_table(recs)
        if s is None:
            print(f"  {name}: no records")
            continue
        print(f"  {name:8s} n={s['n']:3d}  "
              f"aHash (min,q25,med,q75,max) = {s['ahash_pcs']}  "
              f"dHash = {s['dhash_pcs']}")
        print(f"           likely-duplicate hits: {s['n_likely_dup']}/{s['n']} = {s['frac_likely_dup']:.3f}")

    # ---- Show likely-duplicate hits in detail -----------------------------
    print()
    print("=== Likely-duplicate Q×case pairs ===")
    likely_correct = [r for r in correct if r["min_ahash"] <= args.ahash_thresh and r["min_dhash"] <= args.dhash_thresh]
    likely_wrong = [r for r in wrong if r["min_ahash"] <= args.ahash_thresh and r["min_dhash"] <= args.dhash_thresh]
    print(f"Likely duplicates among image-RAG-correct: {len(likely_correct)}")
    for r in likely_correct[:20]:
        print(f"  qid={r['qid']:8s}  body={r['body_system']:14s}  case={r['best_case']:8s}  "
              f"aH={r['min_ahash']:2d} dH={r['min_dhash']:2d}")
        print(f"    q_img:    {r['best_q_path']}")
        print(f"    case_img: {r['best_case_img']}")
    print()
    print(f"Likely duplicates among image-RAG-wrong: {len(likely_wrong)}")
    for r in likely_wrong[:10]:
        print(f"  qid={r['qid']:8s}  body={r['body_system']:14s}  case={r['best_case']:8s}  "
              f"aH={r['min_ahash']:2d} dH={r['min_dhash']:2d}")

    # ---- Interpretation --------------------------------------------------
    print()
    print("=== Interpretation ===")
    if not correct:
        print("  no image-RAG-correct records — cannot compute")
        return
    n_c_likely = sum(1 for r in correct if r["min_ahash"] <= args.ahash_thresh and r["min_dhash"] <= args.dhash_thresh)
    n_w_likely = sum(1 for r in wrong if r["min_ahash"] <= args.ahash_thresh and r["min_dhash"] <= args.dhash_thresh)
    p_c = n_c_likely / len(correct) if correct else 0
    p_w = n_w_likely / len(wrong) if wrong else 0
    print(f"  P(likely-dup | image-RAG correct) = {p_c:.3f}")
    print(f"  P(likely-dup | image-RAG wrong)   = {p_w:.3f}")
    if p_c > 3 * max(p_w, 1e-3):
        print("  → CORRECT set is ≥3× more enriched for near-duplicates. "
              "CONTAMINATION LIKELY — many image-RAG 'wins' come from near-exact retrieval.")
    elif p_c < 0.1 and p_w < 0.1:
        print("  → both <10% likely-duplicate. CONTAMINATION UNLIKELY explanation for the +6pp.")
    else:
        print("  → modest enrichment. Worth manual review of the listed cases.")


if __name__ == "__main__":
    main()
