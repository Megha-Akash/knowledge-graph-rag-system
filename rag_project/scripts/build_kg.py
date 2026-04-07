"""
build_kg.py
CLI entry point for building the knowledge graph.

Usage:
    python scripts/build_kg.py                          # full build
    python scripts/build_kg.py --max-docs 100           # test on 100 docs
    python scripts/build_kg.py --resume                 # resume from checkpoint
    python scripts/build_kg.py --reset                  # rebuild from scratch
    python scripts/build_kg.py --config path/to/config  # custom config
    python scripts/build_kg.py --text "Nolan directed Inception in 2010."  # interactive
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kg_builder.builder_main import build_kg, run_interactive


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a Neo4j knowledge graph from HotpotQA/MuSiQue corpus"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to pipeline config YAML (default: configs/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Extract and write KG from a single text string (interactive mode)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to process (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint if available (default: True)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Clear checkpoint and rebuild from scratch",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.text:
        run_interactive(text=args.text, config_path=args.config)
    else:
        build_kg(
            config_path=args.config,
            max_docs=args.max_docs,
            resume=args.resume,
            reset=args.reset,
        )