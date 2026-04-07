"""
Checkpointer
Tracks which document idx values have been successfully processed.
Enables resuming a crashed or interrupted KG build without reprocessing.

Checkpoint file format:
    {"processed_ids": [0, 1, 2, ...], "total_processed": 3}
"""

import json
import logging
from pathlib import Path
from typing import Set

logger = logging.getLogger(__name__)


class Checkpointer:
    """
    Simple JSON-backed checkpoint store.
    Saves after every batch_size documents to balance safety vs I/O overhead.
    """

    def __init__(self, checkpoint_path: str, batch_size: int = 10):
        self.path = Path(checkpoint_path)
        self.batch_size = batch_size
        self._processed: Set[int] = set()
        self._since_last_save = 0
        self._load()

    def already_processed(self, idx: int) -> bool:
        return idx in self._processed

    def get_processed_ids(self) -> Set[int]:
        return set(self._processed)

    def mark_done(self, idx: int):
        """Mark a document as processed and save checkpoint if batch threshold reached."""
        self._processed.add(idx)
        self._since_last_save += 1
        if self._since_last_save >= self.batch_size:
            self.save()
            self._since_last_save = 0

    def save(self):
        """Force save checkpoint to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(
                {
                    "processed_ids": sorted(self._processed),
                    "total_processed": len(self._processed),
                },
                f,
                indent=2,
            )
        logger.debug(f"Checkpoint saved: {len(self._processed)} docs processed")

    def reset(self):
        """Clear checkpoint — use when starting a fresh build."""
        self._processed = set()
        self._since_last_save = 0
        if self.path.exists():
            self.path.unlink()
        logger.info("Checkpoint reset")

    def _load(self):
        """Load existing checkpoint if present."""
        if not self.path.exists():
            logger.info("No checkpoint found, starting fresh")
            return

        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            self._processed = set(data.get("processed_ids", []))
            logger.info(f"Resumed from checkpoint: {len(self._processed)} docs already processed")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            self._processed = set()