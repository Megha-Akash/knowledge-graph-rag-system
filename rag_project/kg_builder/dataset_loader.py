"""
Dataset Loader
Loads HotpotQA / MuSiQue corpus JSON files and yields clean document dicts.

Expected corpus format (HippoRAG preprocessed):
    [{"idx": 0, "title": "...", "text": "..."}, ...]
"""

import json
import logging
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)


def load_corpus(corpus_path: str, max_docs: Optional[int] = None) -> List[dict]:
    """
    Load full corpus into memory.

    Args:
        corpus_path: Path to corpus JSON file.
        max_docs: If set, only load the first max_docs documents.

    Returns:
        List of document dicts with keys: idx, title, text.
    """
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    logger.info(f"Loading corpus from {corpus_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got {type(data)}")

    if max_docs is not None:
        data = data[:max_docs]

    logger.info(f"Loaded {len(data)} documents")
    return data


def iter_documents(
    corpus_path: str,
    max_docs: Optional[int] = None,
    skip_ids: Optional[set] = None,
) -> Iterator[dict]:
    """
    Iterate over corpus documents, optionally skipping already-processed IDs.

    Args:
        corpus_path: Path to corpus JSON file.
        max_docs: Cap on total documents to yield.
        skip_ids: Set of idx values to skip (for resume/checkpoint logic).

    Yields:
        Document dicts with keys: idx, title, text.
    """
    skip_ids = skip_ids or set()
    docs = load_corpus(corpus_path, max_docs=max_docs)

    skipped = 0
    yielded = 0
    for doc in docs:
        idx = doc.get("idx")
        if idx in skip_ids:
            skipped += 1
            continue
        yield _clean_doc(doc)
        yielded += 1

    logger.info(f"Yielded {yielded} documents, skipped {skipped} (already processed)")


def _clean_doc(doc: dict) -> dict:
    """Normalize a raw corpus entry."""
    return {
        "idx": doc.get("idx"),
        "title": (doc.get("title") or "").strip(),
        "text": (doc.get("text") or "").strip(),
    }