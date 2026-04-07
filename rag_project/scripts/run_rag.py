"""
run_rag.py
CLI entry point for querying the KG-RAG pipeline.

Usage:
    python scripts/run_rag.py --question "Who directed Inception?"
    python scripts/run_rag.py --question "Who directed Inception?" --verbose
    python scripts/run_rag.py --interactive
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_modules.pipeline import RAGPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Query the KG-RAG pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to pipeline config YAML",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question to ask",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print step-by-step pipeline progress",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Interactive mode — ask multiple questions",
    )
    return parser.parse_args()


def run_single(pipeline: RAGPipeline, question: str, verbose: bool):
    result = pipeline.ask(question, verbose=verbose)
    if not verbose:
        print(f"\nAnswer: {result['answer']}")
        print(f"Seeds:  {result['seed_entities']}")
        print("\nContext used:")
        for c in result["context"][:5]:
            print(f"  - {c}")


if __name__ == "__main__":
    args = parse_args()
    pipeline = RAGPipeline(config_path=args.config)

    if args.interactive:
        print("KG-RAG Interactive Mode (type 'quit' to exit)\n")
        while True:
            question = input("Question: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if question:
                run_single(pipeline, question, verbose=args.verbose)
    elif args.question:
        run_single(pipeline, args.question, verbose=args.verbose)
    else:
        print("Provide --question or --interactive. Use --help for usage.")

    pipeline.close()