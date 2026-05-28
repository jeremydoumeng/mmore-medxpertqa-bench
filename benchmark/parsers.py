"""
Answer-letter parsers used by the eval harness.

Four parsers; all extract a single A-J letter from the model's response.
Each handles different output styles:

  - parse_answer_official  : port of MedXpertQA's eval/utils.py
                             `answer_cleansing` (the scoring code used
                             by the public leaderboard). Matches an
                             "ANSWER:" trigger first; falls back to the
                             last A-J letter in the output.
  - parse_answer_strict    : refuses to guess. Requires an explicit
                             "ANSWER: X" / "the answer is X" /
                             "(X) is correct" commit pattern, or a
                             line containing only the letter.
  - parse_answer_lenient   : the harness's pre-fix fallback (last A-J
                             letter anywhere). Kept for cross-method
                             sanity-check, NOT for scoring.
  - parse_answer_boxed     : the CoT prompt produces `\\boxed{X}`;
                             this primary parser for `--prompt-style cot`.
                             Falls back to parse_answer_official.

The reported headline accuracy uses `parse_answer_boxed` (because the
canonical run uses the CoT prompt). Other parsers are logged so cross-
method drift on individual cells can be inspected.
"""

from __future__ import annotations

import re
from typing import Optional


ANSWER_TRIGGER = "ANSWER:"
_AJ_PATTERN = re.compile(r"\b(A|B|C|D|E|F|G|H|I|J)\b")


def parse_answer_official(text: str) -> Optional[str]:
    """Port of MedXpertQA eval/utils.py `answer_cleansing` (few-shot mode).
    Reference: https://github.com/TsinghuaC3I/MedXpertQA/blob/main/eval/utils.py
    """
    if not text:
        return None
    pred = text
    preds = pred.split(ANSWER_TRIGGER)
    answer_flag = len(preds) > 1
    pred = preds[-1]
    for phrase in ("I understand", "A through J", "A through E", "A through D"):
        pred = pred.replace(phrase, "")
    matches = _AJ_PATTERN.findall(pred)
    if not matches:
        return None
    chosen = matches[0] if answer_flag else matches[-1]
    if chosen.endswith("."):
        chosen = chosen[:-1]
    return chosen


def parse_answer_lenient(text: str) -> Optional[str]:
    """Fallback regex; matches last A-J letter anywhere."""
    if not text:
        return None
    stripped = re.sub(r"^\s*(?:A|Answer|ANSWER)\s*[:\-]\s*", "", text, count=1)
    patterns = (
        re.compile(
            r"(?:answer|correct answer|best answer|option|choice)\s*(?:is|:|=)?\s*"
            r"(?:[\(\[\{])?\s*([A-J])\b",
            re.I,
        ),
        re.compile(r"[\(\[\{]\s*([A-J])\s*[\)\]\}]"),
        re.compile(r"(?im)^\s*(?:Answer|A)\s*[:\-]\s*([A-J])\b"),
        re.compile(r"\b([A-J])\s*[)\.]"),
        re.compile(r"\b([A-J])\b"),
    )
    for pat in patterns:
        m = pat.search(stripped)
        if m:
            return m.group(1).upper()
    return None


_ANSWER_PATTERNS = [
    re.compile(r"\bANSWER\s*(?:is|:|=)\s*[\(\[\{]?\s*([A-J])\b", re.I),
    re.compile(
        r"(?:the\s+)?(?:correct|best|right|final)\s+answer\s*(?:is|:|=)?\s*"
        r"[\(\[\{]?\s*([A-J])\b",
        re.I,
    ),
    re.compile(
        r"(?:therefore|hence|thus|so)\s*,?\s*the\s+answer\s+is\s*[\(\[\{]?\s*([A-J])\b",
        re.I,
    ),
    re.compile(
        r"[\(\[\{]\s*([A-J])\s*[\)\]\}]\s+is\s+(?:the\s+)?(?:correct|best|right)\b",
        re.I,
    ),
    re.compile(r"\b([A-J])\s+is\s+(?:the\s+)?(?:correct|best|right)\s+answer\b", re.I),
    re.compile(r"\b([A-J])\.\s+[^:\n]{2,120}:\s*CORRECT\b"),
]
_LINE_ONLY_LETTER = re.compile(r"^\(?\s*([A-J])\s*[\)\.,;:]?\s*$")


def parse_answer_strict(text: str) -> Optional[str]:
    """Refuses to guess. Returns None when no explicit commit is present."""
    if not text:
        return None
    raw = text.strip()
    for pat in _ANSWER_PATTERNS:
        m = pat.search(raw)
        if m:
            return m.group(1).upper()
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    for line in (lines[-1:] if lines else []) + (lines[:1] if lines else []):
        m = _LINE_ONLY_LETTER.match(line)
        if m:
            return m.group(1).upper()
    return None


_BOXED_RE = re.compile(r"\\boxed\{\s*([A-J])\s*\}")


def parse_answer_boxed(text: str) -> Optional[str]:
    """Primary parser for the CoT prompt. Matches `\\boxed{X}`; falls
    back to parse_answer_official if no box found (truncated outputs, etc)."""
    if not text:
        return None
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).upper()
    return parse_answer_official(text)
