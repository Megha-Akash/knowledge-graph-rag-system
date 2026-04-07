"""
KG-RAG Retriever
Retrieves relevant context from Neo4j knowledge graph for a given question.

Improvements over naive RAG:
- Vector similarity search for seed entity finding
- Variable-length multi-hop path traversal
- Cosine similarity reranking of retrieved paths
"""

import logging
from typing import List, Dict, Any

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class KGRetriever:
    """
    Retrieves relevant subgraph context for a question using:
    1. Vector similarity to find seed entities
    2. Variable-length path traversal from seeds
    3. Cosine similarity reranking of retrieved paths
    """

    def __init__(self, driver, embedder: SentenceTransformer,
                 top_k_seeds: int = 5, max_hops: int = 2, top_k_context: int = 15):
        self.driver = driver
        self.embedder = embedder
        self.top_k_seeds = top_k_seeds
        self.max_hops = max_hops
        self.top_k_context = top_k_context

    def retrieve(self, question: str) -> Dict[str, Any]:
        """
        Main retrieval method.

        Args:
            question: Natural language question.

        Returns:
            Dict with context_strings, context_paths, seed_entities.
        """
        # Step 1: embed question
        question_embedding = self.embedder.encode(question, show_progress_bar=False).tolist()

        # Step 2: find seed entities via vector similarity
        seed_entities = self._find_seed_entities(question_embedding)
        if not seed_entities:
            logger.warning("No seed entities found for question")
            return {"context_strings": [], "context_paths": [], "seed_entities": []}

        logger.info(f"Found {len(seed_entities)} seed entities")

        # Step 3: multi-hop traversal from seeds
        paths = self._traverse(seed_entities)
        logger.info(f"Retrieved {len(paths)} raw paths")

        # Step 4: rerank by cosine similarity to question
        ranked_paths = self._rerank(paths, question_embedding)

        # Step 5: format top-k as context strings
        top_paths = ranked_paths[:self.top_k_context]
        context_strings = [self._format_path(p) for p in top_paths]

        return {
            "context_strings": context_strings,
            "context_paths": top_paths,
            "seed_entities": seed_entities,
        }

    def _find_seed_entities(self, question_embedding: List[float]) -> List[Dict]:
        """Find seed entities using vector similarity against node embeddings."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE n.embedding IS NOT NULL AND n.name IS NOT NULL
                RETURN n.name AS name, labels(n) AS labels,
                       n.embedding AS embedding, elementId(n) AS node_id
                LIMIT 5000
                """
            )
            candidates = []
            for record in result:
                embedding = record.get("embedding")
                if embedding:
                    sim = cosine_similarity(question_embedding, embedding)
                    candidates.append({
                        "name": record["name"],
                        "labels": record["labels"],
                        "node_id": record["node_id"],
                        "similarity": sim,
                    })

        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        seeds = candidates[:self.top_k_seeds]

        if seeds:
            logger.info(f"Top seed: {seeds[0]['name']} (sim={seeds[0]['similarity']:.3f})")
        return seeds

    def _traverse(self, seed_entities: List[Dict]) -> List[Dict]:
        """Variable-length path traversal from seed entities."""
        paths = []
        with self.driver.session() as session:
            for seed in seed_entities:
                node_id = seed["node_id"]
                try:
                    result = session.run(
                        f"""
                        MATCH path = (seed)-[*1..{self.max_hops}]-(connected)
                        WHERE elementId(seed) = $node_id
                          AND seed <> connected
                          AND connected.name IS NOT NULL
                        RETURN seed.name AS seed_name,
                               labels(seed) AS seed_labels,
                               connected.name AS connected_name,
                               labels(connected) AS connected_labels,
                               connected.embedding AS connected_embedding,
                               length(path) AS path_length,
                               [r IN relationships(path) | type(r)] AS rel_types
                        LIMIT 30
                        """,
                        {"node_id": node_id},
                    )
                    for record in result:
                        paths.append({
                            "seed_name": record["seed_name"],
                            "seed_labels": record["seed_labels"],
                            "connected_name": record["connected_name"],
                            "connected_labels": record["connected_labels"],
                            "connected_embedding": record["connected_embedding"],
                            "path_length": record["path_length"],
                            "rel_types": record["rel_types"],
                            "seed_entity": seed,
                        })
                except Exception as e:
                    logger.warning(f"Traversal failed for seed '{seed['name']}': {e}")

        return paths

    def _rerank(self, paths: List[Dict], question_embedding: List[float]) -> List[Dict]:
        """Rerank paths by cosine similarity between question and connected node embedding."""
        for path in paths:
            emb = path.get("connected_embedding")
            sim = cosine_similarity(question_embedding, emb) if emb else 0.0
            path_length = path.get("path_length", 1)
            path["relevance_score"] = sim * (1.0 / path_length)

        paths.sort(key=lambda x: x["relevance_score"], reverse=True)
        return paths

    def _format_path(self, path: Dict) -> str:
        """Format a graph path as a readable context string."""
        seed = path.get("seed_name", "")
        connected = path.get("connected_name", "")
        rel_types = path.get("rel_types", [])
        rel_str = " → ".join(rel_types) if rel_types else "RELATED_TO"
        return f"{seed} --[{rel_str}]--> {connected}"