"""Phase 8 — run MedXpertQA-MM eval through PR #281's RAG pipeline.

What this script does, end-to-end, on each question in the n=2000 MM split:

  1. Build a query string = the question's prompt text (the question stem
     + answer choices; the question's own image(s) ride in
     multi_modal_data alongside).
  2. Run mmore-pr281's Retriever against the medpix_pr281 Qdrant
     collection. The retriever returns top-K LangChain Documents whose
     metadata carries `image_paths` (PR #281's contribution).
  3. Format context exactly as RAGPipeline.format_docs does (numbered
     chunk-by-chunk). Aggregate image_paths via PR's
     `aggregate_image_paths()`, load them via `load_images_from_paths()`.
  4. Compose the multimodal user-turn content list:
       [question images, question text+context, retrieved images]
     i.e. exactly the same overall content arrangement as the Phase 7
     hybrid cell, except retrieval modality is text (not ColPali) and
     the retrieved-image attachments come from PR #281's
     image_paths-on-text-chunks pattern.
  5. Generate via `VLLMMultimodalAdapter.invoke_batch_with_images` — one
     vLLM batch over all 2000 prompts per condition (faster than HF
     adapter by ~30x).
  6. Parse the answer (boxed parser primary, official as fallback,
     strict + lenient logged for cross-check). Write per-question JSONL
     and a summary.json in the same schema as Phase 7's Run B / Run C.

Conditions:
  * no-rag         : no retrieval; only question text + question image(s).
  * use_vision-off : text retrieval, only retrieved TEXT passed to VLM
                     (functionally = PR #281's use_vision=False branch =
                     our Phase 7 text-RAG cell, modulo retriever payload).
  * use_vision-on  : text retrieval + retrieved TEXT + attached IMAGES
                     passed to VLM (the PR #281 use_vision=True branch).

Output JSONL schema matches sanity_2324095/results.jsonl so the existing
audit_phase7.py can run paired McNemar across Phase 8 vs Run B/C results.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

# Import mmore from the pinned PR-#281 submodule (third_party/mmore-pr281).
# Honour MMORE_SRC if set (e.g. when mmore is pip-installed into the venv).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MMORE_SRC = os.environ.get(
    "MMORE_SRC",
    str(_REPO_ROOT / "third_party" / "mmore-pr281" / "src"),
)
if _MMORE_SRC and _MMORE_SRC not in sys.path:
    sys.path.insert(0, _MMORE_SRC)

from mmore.index.indexer import DBConfig  # noqa: E402
from mmore.rag.model.vision import (  # noqa: E402
    aggregate_image_paths,
    load_images_from_paths,
)
from mmore.rag.retriever import Retriever, RetrieverConfig  # noqa: E402

# Parsers live next to this script.
sys.path.insert(0, str(Path(__file__).parent))
from parsers import (  # type: ignore  # noqa: E402
    parse_answer_official,
    parse_answer_strict,
    parse_answer_lenient,
    parse_answer_boxed,
)

logging.basicConfig(
    format="[%(asctime)s][eval-pr281] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("eval-pr281")


# Matches the CoT system prompt used by Phase 7 Run B / Run C so the
# baseline (no-rag-CoT, use_vision=False) is directly comparable.
SYSTEM_PROMPT_COT = (
    "You are answering a multiple-choice medical question. The user will "
    "show you one or more medical images and a clinical question, and may "
    "include reference cases (similar medical cases retrieved from a "
    "teaching-file corpus, each with an image and/or a clinical description). "
    "Please reason step by step, and put the final answer in \\boxed{X} "
    "where X is the single letter (A through J) of the chosen option."
)


def load_questions(path: str) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def load_pil(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def format_context(docs) -> str:
    """Mirrors RAGPipeline.format_docs in PR #281 — numbered chunks."""
    return "\n\n".join(
        f"[{doc.metadata.get('rank', i + 1)}] {doc.page_content}"
        for i, doc in enumerate(docs)
    )


def build_query_text(q: Dict[str, Any]) -> str:
    """Query string for the retriever and the LLM. Question stem + options
    inlined (the MM jsonl already has options in q['question']).

    For the retriever we want a query that captures the question topic;
    for the LLM we want the same plus the explicit answer-choice list.
    Keep them identical so the LLM sees the same text the retriever
    matched on (PR #281 also does this — single text channel)."""
    return q["question"]


def build_user_content(
    question_text: str,
    context_text: Optional[str],
    n_question_images: int,
    n_retrieved_images: int,
) -> List[Dict[str, Any]]:
    """Per-question multimodal user-turn content for Qwen2.5-VL.

    Order chosen to match PR #281's pattern *and* keep question images
    in their canonical leading position (Phase 7 used the same ordering,
    so cross-implementation cells are aligned).

    Order: [question images] then text block (question + context) then
    [retrieved images].
    """
    content: List[Dict[str, Any]] = []
    for _ in range(n_question_images):
        content.append({"type": "image"})
    text_parts = [f"Question:\n{question_text}"]
    if context_text:
        text_parts.append(f"Reference cases (text):\n{context_text}")
    content.append({"type": "text", "text": "\n\n".join(text_parts)})
    for _ in range(n_retrieved_images):
        content.append({"type": "image"})
    return content


def build_messages(user_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_COT},
        {"role": "user", "content": user_content},
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", required=True,
                    help="JSONL produced by load_medxpertqa.py --config MM")
    ap.add_argument("--conditions", nargs="+", required=True,
                    choices=["no-rag", "use_vision-off", "use_vision-on"])
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    ap.add_argument("--collection", default="medpix_pr281")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--vlm-tp", type=int, default=4)
    ap.add_argument("--vlm-max-len", type=int, default=32768)
    ap.add_argument("--vlm-max-tokens", type=int, default=1024)
    ap.add_argument("--max-images-per-prompt", type=int, default=12,
                    help="Cap on TOTAL images per prompt (question + "
                         "retrieved). Question images take priority.")
    ap.add_argument("--max-retrieved-images", type=int, default=20,
                    help="PR's max_images_per_request; cap on retrieved "
                         "image attachments. Stays under "
                         "--max-images-per-prompt minus question-image count.")
    ap.add_argument("--results-jsonl", required=True)
    ap.add_argument("--summary-json", required=True)
    args = ap.parse_args()

    log.info("Conditions: %s", args.conditions)
    needs_retrieval = any(c != "no-rag" for c in args.conditions)
    needs_images = "use_vision-on" in args.conditions

    questions = load_questions(args.questions)
    log.info("Loaded %d questions from %s", len(questions), args.questions)

    # ----- Build retriever (only if needed) ---------------------------------
    # Dense/sparse model configs auto-discover from the indexed collection's
    # metadata (Retriever.from_config queries backend.describe_models); no
    # need to pass them here.
    retriever = None
    if needs_retrieval:
        log.info("Building Retriever against Qdrant collection '%s'...",
                 args.collection)
        retriever = Retriever.from_config(RetrieverConfig(
            db=DBConfig(backend="qdrant", uri=args.qdrant_url, name="bench_db"),
            k=args.top_k,
            collection_name=args.collection,
            reranker_model_name=None,
        ))

    # ----- Per-condition: retrieve + build prompts --------------------------
    # We retrieve once per (qid, "with-retrieval") and reuse across
    # use_vision-off and use_vision-on (same retrieval, different context
    # modality). This is faithful to PR #281 — the retrieval doesn't
    # depend on use_vision; only what reaches the LLM does.
    log.info("Building prompts for %d questions × %d conditions...",
             len(questions), len(args.conditions))

    inputs_per_cond: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    records_per_cond: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for q in questions:
        qid = q["id"]
        question_text = q["question"]
        gold = q["label"]
        q_paths: List[str] = list(q.get("image_paths", []))

        # Retrieve once per question (shared by both vision branches)
        docs = []
        context_text = ""
        retrieved_img_paths: List[str] = []
        sources: List[Dict[str, Any]] = []
        if needs_retrieval:
            # LangChain Runnable API. `_get_relevant_documents` requires a
            # callback-manager kwarg we don't set up; invoke() handles it.
            # `collection_name` must be passed in the query dict (else falls
            # back to "my_docs"). `k` is already set on the retriever via
            # RetrieverConfig.k = args.top_k, so it doesn't need to be in
            # the dict (the kwargs path doesn't flow through invoke anyway).
            docs = retriever.invoke({
                "input": question_text,
                "collection_name": args.collection,
            })
            context_text = format_context(docs)
            retrieved_img_paths = aggregate_image_paths(docs)[: args.max_retrieved_images]
            for d in docs:
                sources.append({
                    "u_id": d.metadata.get("id") or "",
                    "rank": d.metadata.get("rank"),
                    "similarity": d.metadata.get("similarity"),
                    "n_attached_images": len(d.metadata.get("image_paths", []) or []),
                })

        for cond in args.conditions:
            if cond == "no-rag":
                user_content = build_user_content(
                    question_text=question_text,
                    context_text=None,
                    n_question_images=len(q_paths),
                    n_retrieved_images=0,
                )
                all_image_paths = q_paths[: args.max_images_per_prompt]
                ctx_chars = 0
                ret_imgs_in_prompt = 0
            elif cond == "use_vision-off":
                user_content = build_user_content(
                    question_text=question_text,
                    context_text=context_text or None,
                    n_question_images=len(q_paths),
                    n_retrieved_images=0,
                )
                all_image_paths = q_paths[: args.max_images_per_prompt]
                ctx_chars = len(context_text or "")
                ret_imgs_in_prompt = 0
            elif cond == "use_vision-on":
                # Image budget: keep question images, fill remainder with
                # retrieved attachments. Question images take precedence.
                remaining = max(0, args.max_images_per_prompt - len(q_paths))
                ret_paths = retrieved_img_paths[:remaining]
                all_image_paths = q_paths + ret_paths
                user_content = build_user_content(
                    question_text=question_text,
                    context_text=context_text or None,
                    n_question_images=len(q_paths),
                    n_retrieved_images=len(ret_paths),
                )
                ctx_chars = len(context_text or "")
                ret_imgs_in_prompt = len(ret_paths)
            else:  # pragma: no cover
                raise ValueError(f"unknown condition: {cond}")

            inputs_per_cond[cond].append({
                "qid": qid,
                "user_content": user_content,
                "image_paths": all_image_paths,
            })
            records_per_cond[cond].append({
                "qid": qid,
                "condition": cond,
                "gold": gold,
                "body_system": q.get("body_system", ""),
                "medical_task": q.get("medical_task", ""),
                "question_type": q.get("question_type", ""),
                "n_question_images": len(q_paths),
                "n_retrieved_images": ret_imgs_in_prompt,
                "ctx_chars": ctx_chars,
                "sources": sources,
            })

    log.info("Per-condition prompt counts: %s",
             {c: len(v) for c, v in inputs_per_cond.items()})

    # ----- Release retriever before loading vLLM ----------------------------
    if retriever is not None:
        log.info("Releasing retriever resources before vLLM init...")
        del retriever
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ----- Load VLM (via PR's vision module + our vLLM adapter) -------------
    log.info("Loading %s via VLLMMultimodalAdapter (tp=%d)...",
             args.vlm_model, args.vlm_tp)
    t0 = time.time()
    from mmore.rag.model.vision import VLLMMultimodalAdapter
    vlm = VLLMMultimodalAdapter(
        model_id=args.vlm_model,
        max_new_tokens=args.vlm_max_tokens,
        tensor_parallel_size=args.vlm_tp,
        max_model_len=args.vlm_max_len,
        max_images_per_prompt=args.max_images_per_prompt,
        gpu_memory_utilization=0.85,
        temperature=0.0,
        dtype="bfloat16",
    )
    log.info("VLM loaded in %.1fs", time.time() - t0)

    # ----- Run each condition through the batch interface -------------------
    # We bypass mmore.rag.pipeline.RAGPipeline's per-question chain
    # invocation here (it would batch=1 through invoke_with_images) for
    # throughput. We're still going through PR #281's retriever,
    # context-aggregation helpers, and his BaseMultimodalLLM abstraction
    # (just via the batched extension method); the retrieval logic and
    # image-attachment logic under test are entirely his.

    Path(args.results_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)

    all_outputs: Dict[str, List[str]] = {}
    for cond in args.conditions:
        log.info("=== Condition: %s  (%d prompts) ===", cond, len(inputs_per_cond[cond]))
        # Render each prompt via the adapter's _build_vllm_input — this
        # uses Qwen2.5-VL's chat template via processor and emits vision
        # tokens at the right positions.
        items: List[Dict[str, Any]] = []
        for inp in inputs_per_cond[cond]:
            # PR's `load_images_from_paths` is what's used in the
            # pipeline's vision branch; reuse it here so any image-load
            # quirks (RGB conversion, missing-file handling) match the
            # framework's behaviour exactly.
            pil_imgs = load_images_from_paths(
                inp["image_paths"],
                max_images=args.max_images_per_prompt,
            )
            # _build_vllm_input expects (text, images) but we have a
            # mixed content list. Build the messages ourselves and skip
            # the helper.
            messages = build_messages(inp["user_content"])
            prompt_str = vlm._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            item: Dict[str, Any] = {"prompt": prompt_str}
            if pil_imgs:
                item["multi_modal_data"] = {"image": pil_imgs}
            items.append(item)

        log.info("Built %d vLLM inputs for cond=%s. Generating...", len(items), cond)
        t0 = time.time()
        outputs = vlm._llm.generate(items, sampling_params=vlm._sampling)
        log.info("cond=%s done in %.1fs", cond, time.time() - t0)
        all_outputs[cond] = [
            (o.outputs[0].text.strip() if o.outputs else "") for o in outputs
        ]

    # ----- Score + write ----------------------------------------------------
    PARSER_NAMES = ("official", "strict", "lenient", "boxed")
    correct_by_cond_parser: Dict[Tuple[str, str], int] = defaultdict(int)
    null_by_cond_parser: Dict[Tuple[str, str], int] = defaultdict(int)
    total_by_cond: Dict[str, int] = defaultdict(int)
    correct_by_cond_system: Dict[Tuple[str, str], int] = defaultdict(int)
    total_by_cond_system: Dict[Tuple[str, str], int] = defaultdict(int)

    with open(args.results_jsonl, "w") as fout:
        for cond in args.conditions:
            for rec, raw_output in zip(records_per_cond[cond], all_outputs[cond]):
                preds = {
                    "official": parse_answer_official(raw_output),
                    "strict":   parse_answer_strict(raw_output),
                    "lenient":  parse_answer_lenient(raw_output),
                    "boxed":    parse_answer_boxed(raw_output),
                }
                corrects = {p: preds[p] == rec["gold"] for p in PARSER_NAMES}
                primary = "boxed"
                sys_key = rec.get("body_system", "")

                total_by_cond[cond] += 1
                correct_by_cond_system[(cond, sys_key)] += int(corrects[primary])
                total_by_cond_system[(cond, sys_key)] += 1
                for p in PARSER_NAMES:
                    correct_by_cond_parser[(cond, p)] += int(corrects[p])
                    if preds[p] is None:
                        null_by_cond_parser[(cond, p)] += 1

                fout.write(json.dumps({
                    "qid": rec["qid"],
                    "condition": cond,
                    "gold": rec["gold"],
                    "predicted": preds[primary],
                    "correct": corrects[primary],
                    "primary_parser": primary,
                    "predicted_official": preds["official"],
                    "predicted_strict":   preds["strict"],
                    "predicted_lenient":  preds["lenient"],
                    "predicted_boxed":    preds["boxed"],
                    "correct_official": corrects["official"],
                    "correct_strict":   corrects["strict"],
                    "correct_lenient":  corrects["lenient"],
                    "correct_boxed":    corrects["boxed"],
                    "raw_output": raw_output,
                    "body_system": sys_key,
                    "medical_task": rec.get("medical_task", ""),
                    "question_type": rec.get("question_type", ""),
                    "n_question_images": rec["n_question_images"],
                    "n_retrieved_images": rec["n_retrieved_images"],
                    "ctx_chars": rec["ctx_chars"],
                    "sources": rec["sources"],
                }, ensure_ascii=False) + "\n")

    primary_parser = "boxed"
    summary = {
        "n_questions": len(questions),
        "vlm_model": args.vlm_model,
        "prompt_style": "cot",
        "primary_parser": primary_parser,
        "framework": "mmore-pr281-port",
        "top_k": args.top_k,
        "by_condition": {
            cond: {
                "total": total_by_cond[cond],
                "correct": correct_by_cond_parser[(cond, primary_parser)],
                "accuracy": (
                    correct_by_cond_parser[(cond, primary_parser)] / total_by_cond[cond]
                ) if total_by_cond[cond] else None,
                "by_parser": {
                    p: {
                        "correct": correct_by_cond_parser[(cond, p)],
                        "accuracy": (
                            correct_by_cond_parser[(cond, p)] / total_by_cond[cond]
                        ) if total_by_cond[cond] else None,
                        "null_pred": null_by_cond_parser[(cond, p)],
                    }
                    for p in PARSER_NAMES
                },
            }
            for cond in args.conditions
        },
        "by_condition_x_body_system": {
            f"{c}::{s}": {
                "correct": correct_by_cond_system[(c, s)],
                "total": total_by_cond_system[(c, s)],
            }
            for (c, s) in sorted(total_by_cond_system)
        },
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2))
    log.info("Wrote results to %s and summary to %s",
             args.results_jsonl, args.summary_json)
    for cond, stats in summary["by_condition"].items():
        bp = stats["by_parser"]
        log.info(
            "  %-16s  n=%d  boxed=%.4f  official=%.4f  strict=%.4f  lenient=%.4f  (null boxed=%d)",
            cond, stats["total"],
            bp["boxed"]["accuracy"],
            bp["official"]["accuracy"],
            bp["strict"]["accuracy"],
            bp["lenient"]["accuracy"],
            bp["boxed"]["null_pred"],
        )


if __name__ == "__main__":
    main()
