"""Phase 8 — index MedPix-2.0 as mmore documents through PR #281's pipeline.

PR #281's data model is `documents-with-attachments`: a MultimodalSample
has text + a list of MultimodalRawInput modalities, some of which are
`image` attachments. The ported `Indexer._index_documents` then persists
the `image_paths` field (JSON-encoded list of attachment values) into
the Qdrant payload, where it can be retrieved alongside the text chunk.

MedPix's natural unit is the *case* (671 unique cases, ~3 images per
case, one consolidated case text). Mapping:

    case  -->  MultimodalSample(
                  text       = consolidated_case_text,
                  modalities = [MultimodalRawInput(type="image", value=img_path)
                                for img_path in case_image_paths],
                  metadata   = {"u_id": u_id, ...},
                  id         = u_id,
                  document_id= u_id,
              )

This script reads `medpix_cases.parquet` and `medpix_case_text.parquet`
(built in Phase 7) and indexes the cases via mmore-pr281's Indexer into
a fresh Qdrant collection (`medpix_pr281`).

Usage:
    python bench/eval-pr281/index_medpix_as_mmore_docs.py \\
        --cases-parquet $SCRATCH/medpix2/medpix_cases.parquet \\
        --text-parquet $SCRATCH/medpix2/medpix_case_text.parquet \\
        --qdrant-url http://127.0.0.1:6333 \\
        --collection medpix_pr281
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Import mmore from the pinned PR-#281 submodule (third_party/mmore-pr281).
# Honour MMORE_SRC if set (e.g. when mmore is pip-installed into the venv).
import os
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MMORE_SRC = os.environ.get(
    "MMORE_SRC",
    str(_REPO_ROOT / "third_party" / "mmore-pr281" / "src"),
)
if _MMORE_SRC and _MMORE_SRC not in sys.path:
    sys.path.insert(0, _MMORE_SRC)

from mmore.index.indexer import DBConfig, Indexer, IndexerConfig  # noqa: E402
from mmore.rag.model.dense.base import DenseModelConfig  # noqa: E402
from mmore.rag.model.sparse.base import SparseModelConfig  # noqa: E402
from mmore.type import MultimodalRawInput, MultimodalSample  # noqa: E402

logging.basicConfig(
    format="[%(asctime)s][medpix-pr281-idx] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("medpix-pr281-idx")


DEFAULT_DENSE = os.environ.get("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_SPARSE = os.environ.get("SPARSE_MODEL", "splade")


def build_samples(
    cases_parquet: str, text_parquet: str
) -> List[MultimodalSample]:
    """One MultimodalSample per unique case. Text from text_parquet (case-text
    consolidated chunk). Modalities = list of image attachments from
    cases_parquet for that u_id."""
    cases_df = pd.read_parquet(cases_parquet)
    text_df = pd.read_parquet(text_parquet)
    log.info(
        "Loaded %d case-image rows (%d unique u_ids) and %d case-text rows",
        len(cases_df),
        cases_df["u_id"].nunique(),
        len(text_df),
    )

    # u_id -> [image_path, ...] sorted by image_idx for deterministic ordering
    images_by_uid: Dict[str, List[str]] = defaultdict(list)
    for _, row in cases_df.sort_values(["u_id", "image_idx"]).iterrows():
        p = str(row["image_path"])
        if Path(p).exists():
            images_by_uid[str(row["u_id"])].append(p)

    # u_id -> case_text  (the text_parquet uses pdf_path=u_id, page_number=0,
    # text=case_text per Phase 7's construction)
    text_by_uid: Dict[str, str] = dict(
        zip(text_df["pdf_path"].astype(str), text_df["text"])
    )

    samples: List[MultimodalSample] = []
    missing_text = 0
    no_images = 0
    for uid in sorted(text_by_uid):
        text = text_by_uid[uid]
        if not text or not text.strip():
            missing_text += 1
            continue
        image_paths = images_by_uid.get(uid, [])
        if not image_paths:
            no_images += 1
            # Still index it — the PR's use_vision=False path still wants the
            # text chunk; use_vision=True will simply have no attachments
            # for this case. Don't drop these from the corpus.
        modalities = [
            MultimodalRawInput(type="image", value=ip) for ip in image_paths
        ]
        samples.append(
            MultimodalSample(
                text=text,
                modalities=modalities,
                metadata={
                    "u_id": uid,
                    "n_images": len(image_paths),
                },
                id=uid,
                document_id=uid,
            )
        )

    log.info(
        "Built %d samples (skipped %d with empty text; %d had no on-disk images)",
        len(samples),
        missing_text,
        no_images,
    )
    return samples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases-parquet", required=True)
    ap.add_argument("--text-parquet", required=True)
    ap.add_argument("--qdrant-url", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dense-model", default=DEFAULT_DENSE)
    ap.add_argument("--sparse-model", default=DEFAULT_SPARSE)
    args = ap.parse_args()

    samples = build_samples(args.cases_parquet, args.text_parquet)

    # Build the Indexer with Qdrant backend
    db_cfg = DBConfig(backend="qdrant", uri=args.qdrant_url, name="bench_db")
    dense_cfg = DenseModelConfig(model_name=args.dense_model)
    sparse_cfg = SparseModelConfig(model_name=args.sparse_model)
    indexer_cfg = IndexerConfig(
        db=db_cfg,
        dense_model=dense_cfg,
        sparse_model=sparse_cfg,
    )
    log.info("Building Indexer (dense=%s, sparse=%s, backend=qdrant)...",
             args.dense_model, args.sparse_model)
    indexer = Indexer.from_config(indexer_cfg)

    log.info("Indexing %d samples into Qdrant collection '%s'...",
             len(samples), args.collection)
    indexer.index_documents(
        documents=samples,
        collection_name=args.collection,
        batch_size=args.batch_size,
    )

    # Spot-check: query the collection for a sample and confirm image_paths
    # land in the Qdrant payload as expected by PR #281's retriever.
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(args.qdrant_url)
        scroll_pts, _ = client.scroll(
            collection_name=args.collection, limit=3, with_payload=True
        )
        log.info("Spot-check — first 3 points in collection:")
        for pt in scroll_pts:
            payload = pt.payload or {}
            log.info(
                "  id=%s  text[:60]=%r  image_paths(raw)=%r",
                payload.get("id") or payload.get("u_id"),
                (payload.get("text") or "")[:60],
                payload.get("image_paths"),
            )
    except Exception as e:
        log.warning("Spot-check failed (non-fatal): %s", e)

    indexer.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
