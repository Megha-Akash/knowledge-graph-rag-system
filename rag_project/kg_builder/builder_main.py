"""
Builder Main
Orchestrates the full KG build pipeline:
    load corpus → extract → write to Neo4j → checkpoint
"""

import logging
import time
from typing import Optional

import yaml

from .dataset_loader import iter_documents
from .extractor import KGExtractor
from .neo4j_writer import Neo4jWriter
from .checkpointer import Checkpointer

logger = logging.getLogger(__name__)


def build_kg(config_path: str = "configs/pipeline_config.yaml",
             max_docs: Optional[int] = None,
             resume: bool = True,
             reset: bool = False):
    """
    Main entry point for KG building.

    Args:
        config_path: Path to pipeline YAML config.
        max_docs: Override max_docs from config (useful for quick tests).
        resume: If True, skip already-processed documents via checkpoint.
        reset: If True, clear checkpoint and rebuild from scratch.
    """
    config = _load_config(config_path)
    _setup_logging()

    kg_cfg = config["kg_builder"]
    llm_cfg = config["llm"]
    neo4j_cfg = config["neo4j"]
    data_cfg = config["data"]
    embed_cfg = config["embedder"]

    effective_max_docs = max_docs if max_docs is not None else kg_cfg.get("max_docs")

    # Init components
    checkpointer = Checkpointer(
        checkpoint_path=kg_cfg["checkpoint_path"],
        batch_size=kg_cfg.get("batch_size", 10),
    )

    if reset:
        checkpointer.reset()
        logger.info("Reset requested — rebuilding from scratch")

    extractor = KGExtractor(
        model=llm_cfg["model"],
        endpoint=llm_cfg["endpoint"],
        temperature=llm_cfg.get("temperature", 0.1),
        max_tokens=llm_cfg.get("max_tokens", 1500),
    )

    writer = Neo4jWriter(
        uri=neo4j_cfg["uri"],
        user=neo4j_cfg["user"],
        password=neo4j_cfg["password"],
        embedder_model=embed_cfg.get("model", "all-MiniLM-L6-v2"),
    )

    skip_ids = checkpointer.get_processed_ids() if resume else set()
    logger.info(f"Skipping {len(skip_ids)} already-processed documents")

    # Main build loop
    total = 0
    failed = 0
    start = time.time()

    try:
        for doc in iter_documents(
            corpus_path=data_cfg["corpus_path"],
            max_docs=effective_max_docs,
            skip_ids=skip_ids,
        ):
            idx = doc["idx"]
            logger.info(f"Processing doc idx={idx}: {doc['title'][:60]}")

            try:
                extracted = extractor.extract(doc)
                writer.write_document(doc, extracted)
                checkpointer.mark_done(idx)
                total += 1

                if total % 50 == 0:
                    elapsed = time.time() - start
                    rate = total / elapsed
                    logger.info(f"Progress: {total} docs | {rate:.1f} docs/sec | failed: {failed}")

            except Exception as e:
                logger.error(f"Failed on idx={idx}: {e}")
                failed += 1

    finally:
        # Always save checkpoint on exit
        checkpointer.save()
        writer.close()

    elapsed = time.time() - start
    logger.info(
        f"Build complete: {total} processed, {failed} failed, "
        f"{elapsed:.1f}s total ({total/elapsed:.1f} docs/sec)"
    )


def run_interactive(text: str, config_path: str = "configs/pipeline_config.yaml"):
    """
    Extract and write KG from a single text string.
    Useful for quick testing without running the full pipeline.

    Args:
        text: Raw text to extract from.
        config_path: Path to pipeline YAML config.
    """
    config = _load_config(config_path)
    _setup_logging()

    llm_cfg = config["llm"]
    neo4j_cfg = config["neo4j"]
    embed_cfg = config["embedder"]

    extractor = KGExtractor(
        model=llm_cfg["model"],
        endpoint=llm_cfg["endpoint"],
        temperature=llm_cfg.get("temperature", 0.1),
        max_tokens=llm_cfg.get("max_tokens", 1500),
    )

    writer = Neo4jWriter(
        uri=neo4j_cfg["uri"],
        user=neo4j_cfg["user"],
        password=neo4j_cfg["password"],
        embedder_model=embed_cfg.get("model", "all-MiniLM-L6-v2"),
    )

    doc = {"idx": -1, "title": "", "text": text}

    print("\n--- Extracting from text ---")
    extracted = extractor.extract(doc)

    print(f"Entities ({len(extracted['entities'])}):")
    for e in extracted["entities"]:
        print(f"  [{e['label']}] {e['properties'].get('name')}")

    print(f"Relations ({len(extracted['relations'])}):")
    for r in extracted["relations"]:
        print(f"  {r['start_entity']['name']} --[{r['label']}]--> {r['end_entity']['name']}")

    print(f"Topics: {extracted['topics']}")

    writer.write_document(doc, extracted)
    writer.close()
    print("\nWritten to Neo4j.")


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )