"""Gold-coverage subset audit.

Definition (operational): a MedXpertQA-MM question is "gold-covered" by
MedPix-2.0 if its CORRECT option text matches the title or
case_diagnosis of at least one MedPix case under either:
  - case-insensitive substring containment, OR
  - rapidfuzz token-set partial ratio >= 90 (catches abbreviation/
    short-form variants like "NSCLC" ↔ "non-small cell lung cancer")

This is a necessary condition for retrieval to help: if the answer
concept doesn't exist in the corpus, no amount of better retrieval can
fix that. It is NOT sufficient — a covered question can still fail if
the retriever can't surface the matching case from the query text.

For each Phase 7 / Phase 8 results.jsonl, re-score on the covered
subset only: accuracy + Wilson CI per condition, pairwise McNemar +
Holm correction against no-RAG.

Usage:
    python gold_coverage_audit.py
"""

from __future__ import annotations
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# All inputs are CLI args (see main()); no path constants baked in.


def normalise(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_substring(needle: str, haystack: str) -> bool:
    n, h = normalise(needle), normalise(haystack)
    if not n or not h:
        return False
    return n in h


_STOPWORDS = {"the", "a", "of", "and", "or", "in", "on", "to", "with", "by", "for"}


def _tokens(s: str) -> set:
    return {t for t in normalise(s).split() if t not in _STOPWORDS and len(t) >= 3}


def token_set_ratio(a: str, b: str) -> int:
    """Pure-Python token-set ratio: percentage of needle-side tokens that
    appear in haystack-side. Matches rapidfuzz's behaviour closely enough
    for the substring/concept matching we need (no Levenshtein on full
    strings, just bag-of-tokens overlap). Returns 0-100."""
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta:
        return 0
    intersection = ta & tb
    # Symmetric overlap normalised to the smaller token-set, mirroring
    # the rapidfuzz token-set-partial-ratio behaviour.
    smaller = min(len(ta), len(tb)) or 1
    return int(100 * len(intersection) / smaller)


def is_fuzzy_match(needle: str, haystack: str, threshold: int = 90) -> bool:
    n, h = normalise(needle), normalise(haystack)
    if not n or not h:
        return False
    return token_set_ratio(n, h) >= threshold


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def mcnemar(b: int, c: int) -> Tuple[float, float]:
    if b + c == 0:
        return 0.0, 1.0
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


def build_coverage_set(questions_jsonl: str, cases_parquet: str) -> Dict[str, dict]:
    """Returns qid -> {covered, match_strategy, matched_uid, matched_diagnosis, gold_text}"""
    questions = [json.loads(l) for l in open(questions_jsonl)]
    cases = pd.read_parquet(cases_parquet).drop_duplicates("u_id")

    diagnoses = []
    for _, r in cases.iterrows():
        title = str(r["title"] or "")
        case_dx = str(r["case_diagnosis"] or "")
        # Use the union of title + case_diagnosis as the case-side concept
        # space. Title is usually the primary diagnosis; case_diagnosis is
        # sometimes the same, sometimes more specific.
        diagnoses.append((str(r["u_id"]), title, case_dx))

    coverage: Dict[str, dict] = {}
    for q in questions:
        qid = q["id"]
        gold_text = q["options"][q["label"]]
        match_strategy: Optional[str] = None
        matched_uid: Optional[str] = None
        matched_dx: Optional[str] = None

        # Skip non-matchable gold answers (single letters, numeric values,
        # short labels). ~111/2000 questions have gold_text <= 3 chars —
        # typically value-choice (e.g. "8.8") or matching-format
        # (e.g. "C") items where no medical concept is in play.
        normalised_gold = normalise(gold_text)
        if len(normalised_gold) < 4 or len(_tokens(gold_text)) == 0:
            coverage[qid] = {
                "covered": False,
                "match_strategy": "non-matchable-gold",
                "matched_uid": None,
                "matched_dx": None,
                "gold_text": gold_text,
                "body_system": q.get("body_system", ""),
            }
            continue

        # 1) strict substring (case-insensitive, punctuation-stripped)
        for uid, title, case_dx in diagnoses:
            if is_substring(gold_text, title) or is_substring(gold_text, case_dx):
                match_strategy = "substring"
                matched_uid = uid
                matched_dx = case_dx if case_dx else title
                break

        # 2) fall back to token-set fuzzy match
        if match_strategy is None:
            best = (0, None, None)
            for uid, title, case_dx in diagnoses:
                s1 = token_set_ratio(gold_text, title)
                s2 = token_set_ratio(gold_text, case_dx)
                s = max(s1, s2)
                if s > best[0]:
                    best = (s, uid, case_dx if s2 >= s1 else title)
            if best[0] >= 90:
                match_strategy = f"fuzzy{best[0]}"
                matched_uid = best[1]
                matched_dx = best[2]

        coverage[qid] = {
            "covered": match_strategy is not None,
            "match_strategy": match_strategy,
            "matched_uid": matched_uid,
            "matched_dx": matched_dx,
            "gold_text": gold_text,
            "body_system": q.get("body_system", ""),
        }
    return coverage


def load_results(path: str) -> Dict[Tuple[str, str], dict]:
    """(qid, condition) -> {correct, body_system}"""
    out: Dict[Tuple[str, str], dict] = {}
    for line in open(path):
        r = json.loads(line)
        # CoT runs use correct_boxed; fall back to correct otherwise.
        c = r.get("correct_boxed", r.get("correct"))
        out[(r["qid"], r["condition"])] = {
            "correct": bool(c),
            "body_system": r.get("body_system", ""),
        }
    return out


def analyse_run(label: str, results_path: str, coverage: Dict[str, dict]) -> None:
    res = load_results(results_path)
    qids = sorted({q for (q, _) in res})
    conds_in_run = sorted({c for (_, c) in res})

    covered_qids = [q for q in qids if coverage.get(q, {}).get("covered")]
    uncov_qids = [q for q in qids if not coverage.get(q, {}).get("covered")]

    print(f"\n========== {label} ==========")
    print(f"   file: {Path(results_path).name}")
    print(f"   n_total={len(qids)}  n_covered={len(covered_qids)}  n_uncov={len(uncov_qids)}")

    def acc(qid_set, cond):
        k = sum(1 for q in qid_set if res.get((q, cond), {}).get("correct"))
        n = sum(1 for q in qid_set if (q, cond) in res)
        return k, n

    print(f"\n{'condition':18s}  {'subset':14s}  {'k':>4s}/{'n':>4s}  {'acc':>6s}  {'95% Wilson CI':<18s}")
    by_cond_covered_acc: Dict[str, float] = {}
    for c in conds_in_run:
        k, n = acc(qids, c)
        p, lo, hi = wilson_ci(k, n)
        print(f"  {c:16s}  {'all (control)':14s}  {k:4d}/{n:4d}  {p:.4f}  [{lo:.4f}, {hi:.4f}]")
        k, n = acc(covered_qids, c)
        p, lo, hi = wilson_ci(k, n)
        by_cond_covered_acc[c] = p
        print(f"  {c:16s}  {'covered':14s}  {k:4d}/{n:4d}  {p:.4f}  [{lo:.4f}, {hi:.4f}]")
        k, n = acc(uncov_qids, c)
        p, lo, hi = wilson_ci(k, n)
        print(f"  {c:16s}  {'uncovered':14s}  {k:4d}/{n:4d}  {p:.4f}  [{lo:.4f}, {hi:.4f}]")

    # Paired McNemar on covered subset only — same family as the
    # full-split audit, Holm-corrected.
    rag_conds = [c for c in conds_in_run if c != "no-rag"]
    print(f"\n  Pairwise McNemar on COVERED subset only (vs no-rag, Holm-{len(rag_conds)}):")
    rows = []
    for cond in rag_conds:
        b = c_ = 0
        for q in covered_qids:
            A = res.get((q, "no-rag"), {}).get("correct")
            B = res.get((q, cond), {}).get("correct")
            if A is None or B is None:
                continue
            if A and not B: b += 1
            elif B and not A: c_ += 1
        chi2, p = mcnemar(b, c_)
        delta = (by_cond_covered_acc[cond] - by_cond_covered_acc["no-rag"]) * 100
        rows.append((cond, b, c_, chi2, p, delta))
    pvals = [r[4] for r in rows]
    holm_p = holm(pvals)
    print(f"  {'cond':18s} {'b':>4s} {'c':>4s} {'chi2':>6s}  {'p_raw':>8s}  {'p_holm':>8s}  {'Δ vs no-rag (pp)':>20s}")
    for (cond, b, c_, chi2, p, delta), p_adj in zip(rows, holm_p):
        sig = " *" if p_adj < 0.05 else ""
        print(f"  {cond:18s} {b:4d} {c_:4d} {chi2:6.2f}  {p:.4f}  {p_adj:.4f}{sig:2s}  {delta:+19.2f}")


def coverage_summary(coverage: Dict[str, dict]) -> None:
    total = len(coverage)
    covered = sum(1 for v in coverage.values() if v["covered"])
    strats = defaultdict(int)
    for v in coverage.values():
        strats[v["match_strategy"] or "no-match"] += 1
    print(f"=== gold-coverage subset ===")
    print(f"  total questions: {total}")
    print(f"  covered:         {covered}  ({covered/total:.1%})")
    print(f"  match strategies: {dict(strats)}")
    print()

    # Body-system breakdown
    sys_total = defaultdict(int)
    sys_cov = defaultdict(int)
    for v in coverage.values():
        sys_total[v["body_system"]] += 1
        if v["covered"]:
            sys_cov[v["body_system"]] += 1
    print(f"  body-system breakdown:")
    print(f"  {'body_system':25s} {'covered':>10s}/{'total':<6s}  pct")
    for sysm in sorted(sys_total, key=lambda s: -sys_total[s]):
        n = sys_total[sysm]
        k = sys_cov[sysm]
        print(f"  {sysm:25s} {k:>10d}/{n:<6d}  {k/n:.1%}")

    # Sample 5 covered + 5 uncovered for sanity inspection
    cov_items = [(q, v) for q, v in coverage.items() if v["covered"]]
    uncov_items = [(q, v) for q, v in coverage.items() if not v["covered"]]
    print(f"\n  sample covered (gold_text → matched MedPix concept):")
    for q, v in cov_items[:7]:
        print(f"    {q}: {v['gold_text'][:55]!r}  →  [{v['matched_uid']}] {v['matched_dx'][:55]!r}  ({v['match_strategy']})")
    print(f"\n  sample uncovered (gold_text — not in MedPix):")
    for q, v in uncov_items[:7]:
        print(f"    {q}: {v['gold_text'][:80]!r}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-jsonl", required=True,
                    help="An eval results.jsonl produced by benchmark/run_eval.py")
    ap.add_argument("--questions-jsonl", required=True,
                    help="MedXpertQA-MM jsonl (typically the 2000-q split used by the eval)")
    ap.add_argument("--cases-parquet", required=True,
                    help="MedPix cases.parquet built by data/load_medpix.py")
    ap.add_argument("--label", default=None,
                    help="Optional label printed in the report (default: file path)")
    args = ap.parse_args()

    coverage = build_coverage_set(args.questions_jsonl, args.cases_parquet)
    coverage_summary(coverage)
    analyse_run(args.label or args.results_jsonl, args.results_jsonl, coverage)


if __name__ == "__main__":
    main()
