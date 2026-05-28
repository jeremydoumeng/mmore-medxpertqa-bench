"""
Within-results-file paired McNemar audit.

For an eval results.jsonl produced by benchmark/run_eval.py:

  1. Accuracy + 95% Wilson CI per condition (boxed parser primary).
  2. Pairwise McNemar test (each non-baseline condition vs baseline),
     Holm-corrected over the comparison family.
  3. Per-body-system McNemar (baseline vs the last non-baseline cond)
     with Holm correction over the 12 body systems.
  4. Discordant-pair head-counts for every pair (so the "real n" of
     each comparison is visible).

Usage:
    python audit/mcnemar.py --results-jsonl results/eval_<JOBID>/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def mcnemar(b: int, c: int) -> Tuple[float, float]:
    """Continuity-corrected McNemar. Returns (chi2, two-sided p)."""
    if b + c == 0:
        return (0.0, 1.0)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2))
    return chi2, p


def holm(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        adj_p = min(1.0, pvals[i] * (m - rank))
        running_max = max(running_max, adj_p)
        adj[i] = running_max
    return adj


def load_results(path: str, correct_field: str) -> Dict[Tuple[str, str], dict]:
    """(qid, condition) -> {correct, body_system}"""
    out: Dict[Tuple[str, str], dict] = {}
    for line in open(path):
        r = json.loads(line)
        c = r.get(correct_field, r.get("correct"))
        out[(r["qid"], r["condition"])] = {
            "correct": bool(c),
            "body_system": r.get("body_system", ""),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-jsonl", required=True)
    ap.add_argument("--correct-field", default="correct_boxed",
                    help="Field to read for correctness. correct_boxed for "
                         "CoT runs (default); correct_official otherwise.")
    ap.add_argument("--baseline", default="no-rag",
                    help="Condition used as the McNemar baseline.")
    args = ap.parse_args()

    res = load_results(args.results_jsonl, args.correct_field)
    qids = sorted({q for (q, _) in res})
    conds = sorted({c for (_, c) in res})
    print(f"=== {Path(args.results_jsonl).resolve()} ===")
    print(f"  n_qids={len(qids)}  conditions={conds}")
    if args.baseline not in conds:
        raise SystemExit(f"--baseline '{args.baseline}' not in conditions {conds}")

    # 1. Accuracy + Wilson CI per condition
    print(f"\n=== Accuracy + 95% Wilson CI ===")
    print(f"{'cond':18s} {'k':>4s}/{'n':>4s}  {'acc':>6s}  Wilson 95% CI")
    for c in conds:
        k = sum(1 for q in qids if res.get((q, c), {}).get("correct"))
        n = sum(1 for q in qids if (q, c) in res)
        p, lo, hi = wilson_ci(k, n)
        print(f"  {c:16s} {k:4d}/{n:4d}  {p:.4f}  [{lo:.4f}, {hi:.4f}]")

    # 2. Pairwise McNemar vs baseline, Holm-corrected
    rag_conds = [c for c in conds if c != args.baseline]
    print(f"\n=== Pairwise McNemar (vs {args.baseline}, Holm-{len(rag_conds)}) ===")
    print(f"{'cond':18s} {'b':>4s} {'c':>4s} {'chi2':>6s}  {'p_raw':>8s}  {'p_holm':>8s}  {'Δ vs base (pp)':>16s}")
    base_n = sum(1 for q in qids if (q, args.baseline) in res)
    base_k = sum(1 for q in qids if res.get((q, args.baseline), {}).get("correct"))
    base_p = base_k / base_n if base_n else 0.0
    rows = []
    for cond in rag_conds:
        b = c_ = 0
        for q in qids:
            A = res.get((q, args.baseline), {}).get("correct")
            B = res.get((q, cond), {}).get("correct")
            if A is None or B is None:
                continue
            if A and not B: b += 1
            elif B and not A: c_ += 1
        chi2, p = mcnemar(b, c_)
        cond_n = sum(1 for q in qids if (q, cond) in res)
        cond_k = sum(1 for q in qids if res.get((q, cond), {}).get("correct"))
        delta = ((cond_k / cond_n) - base_p) * 100 if cond_n else 0.0
        rows.append((cond, b, c_, chi2, p, delta))
    pvals = [r[4] for r in rows]
    holm_p = holm(pvals)
    for (cond, b, c_, chi2, p, delta), p_adj in zip(rows, holm_p):
        sig = " *" if p_adj < 0.05 else ""
        print(f"  {cond:16s} {b:4d} {c_:4d} {chi2:6.2f}  {p:.4f}  {p_adj:.4f}{sig:2s}  {delta:+15.2f}")

    # 3. Per body-system McNemar (baseline vs last RAG cond), Holm over systems
    target_cond = rag_conds[-1] if rag_conds else None
    if target_cond:
        print(f"\n=== Per body_system: McNemar ({args.baseline} vs {target_cond}), Holm-corrected ===")
        body_systems = sorted({
            res[(q, args.baseline)].get("body_system", "")
            for q in qids
            if (q, args.baseline) in res
        })
        body_systems = [b for b in body_systems if b]
        sys_rows = []
        for bs in body_systems:
            b = c_ = 0
            n = 0
            for q in qids:
                ent = res.get((q, args.baseline), {})
                if ent.get("body_system") != bs:
                    continue
                n += 1
                A = ent.get("correct")
                B = res.get((q, target_cond), {}).get("correct")
                if A and not B: b += 1
                elif B and not A: c_ += 1
            chi2, p = mcnemar(b, c_)
            sys_rows.append((bs, n, b, c_, chi2, p))
        pvals = [r[5] for r in sys_rows]
        holm_p = holm(pvals)
        print(f"{'body_system':22s} {'n':>3s} {'b':>3s} {'c':>3s} {'chi2':>6s}  {'p_raw':>8s}  {'p_holm':>8s}")
        for (bs, n, b, c_, chi2, p), p_adj in zip(sys_rows, holm_p):
            sig = " *" if p_adj < 0.05 else ""
            print(f"  {bs:20s} {n:3d} {b:3d} {c_:3d} {chi2:6.2f}  {p:.4f}  {p_adj:.4f}{sig}")

    # 4. Discordant-pair head-counts for every pair
    print(f"\n=== Discordant-pair head-counts ===")
    for i, a in enumerate(conds):
        for bc in conds[i + 1 :]:
            b = c_ = 0
            for q in qids:
                A = res.get((q, a), {}).get("correct")
                B = res.get((q, bc), {}).get("correct")
                if A is None or B is None:
                    continue
                if A and not B: b += 1
                elif B and not A: c_ += 1
            net = c_ - b
            print(f"  {a:18s} vs {bc:18s}  b+c={b+c_:4d}  net_({bc}>{a})={net:+4d}")


if __name__ == "__main__":
    main()
