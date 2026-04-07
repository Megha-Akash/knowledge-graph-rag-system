"""
RAG Pipeline
Orchestrates KG retrieval and answer generation.

Flow:
    question
      KGRetriever: find seed entities via vector similarity,
                     traverse graph, rerank by cosine similarity
      AnswerGenerator: generate answer from top-k context
      return answer + evidence
"""

import logging

import yaml
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from .retriever import KGRetriever
from .generator import AnswerGenerator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end KG-RAG pipeline.
    Retrieves relevant subgraph context and generates grounded answers.
    """

    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()

        neo4j_cfg = self.config["neo4j"]
        llm_cfg = self.config["llm"]
        embed_cfg = self.config["embedder"]
        rag_cfg = self.config.get("rag", {})

        self.driver = GraphDatabase.driver(
            neo4j_cfg["uri"], auth=(neo4j_cfg["user"], neo4j_cfg["password"])
        )
        self.embedder = SentenceTransformer(embed_cfg.get("model", "all-MiniLM-L6-v2"))

        self.retriever = KGRetriever(
            driver=self.driver,
            embedder=self.embedder,
            top_k_seeds=rag_cfg.get("top_k_seeds", 5),
            max_hops=rag_cfg.get("max_hops", 2),
            top_k_context=rag_cfg.get("top_k_context", 15),
        )

        self.generator = AnswerGenerator(
            model=llm_cfg["model"],
            endpoint=llm_cfg["endpoint"],
            temperature=llm_cfg.get("temperature", 0.3),
            max_tokens=llm_cfg.get("max_tokens", 512),
        )

        logger.info("RAG pipeline initialized")

    def ask(self, question: str, verbose: bool = False) -> dict:
        """
        Ask a question and get a grounded answer from the knowledge graph.

        Args:
            question: Natural language question.
            verbose: Print step-by-step progress.

        Returns:
            Dict with answer, context, seed_entities.
        """
        if verbose:
            print(f"\n=== KG-RAG Pipeline ===")
            print(f"Question: {question}")

        # Step 1: retrieve context from KG
        if verbose:
            print("\n[1] Retrieving context from knowledge graph...")
        retrieval = self.retriever.retrieve(question)
        context_strings = retrieval["context_strings"]

        if verbose:
            print(f"    Seed entities: {[e['name'] for e in retrieval['seed_entities']]}")
            print(f"    Retrieved {len(context_strings)} context paths")
            if context_strings:
                print("    Top 3 context paths:")
                for c in context_strings[:3]:
                    print(f"      - {c}")

        if not context_strings:
            return {
                "answer": "No relevant context found in knowledge graph.",
                "context": [],
                "seed_entities": [],
            }

        # Step 2: generate answer
        if verbose:
            print("\n[2] Generating answer...")
        answer = self.generator.generate(question, context_strings[:10])

        if verbose:
            print(f"\n=== Answer ===\n{answer}")

        return {
            "answer": answer,
            "context": context_strings[:10],
            "seed_entities": [e["name"] for e in retrieval["seed_entities"]],
        }

    def close(self):
        self.driver.close()

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )