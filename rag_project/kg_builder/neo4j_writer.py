"""
Neo4j Writer
Writes extracted KG data to Neo4j with embeddings.
Based on the original FZI graph_utils.py, cleaned up for rag_project.
"""

import logging
from typing import Dict, Any, List

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def flatten_properties(props: dict, parent_key: str = "", sep: str = "_") -> dict:
    flat = {}
    for key, value in props.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flat.update(flatten_properties(value, parent_key=new_key, sep=sep))
        elif value is not None:
            flat[new_key] = value
    return flat


def create_node_text_for_embedding(label: str, props: dict) -> str:
    """Create a text representation of a node for embedding generation."""
    parts = []
    for key in ["name", "title", "field", "birth_date"]:
        if props.get(key):
            parts.append(f"{key.replace('_', ' ').title()}: {props[key]}")
    parts.append(f"Type: {label}")
    for key, value in props.items():
        if key not in ["name", "title", "field", "birth_date"] and value:
            parts.append(f"{key.title()}: {value}")
    return ". ".join(parts)


class Neo4jWriter:
    """
    Handles all Neo4j write operations for the KG builder.
    Writes entities with embeddings, relationships, and topics.
    """

    def __init__(self, uri: str, user: str, password: str, embedder_model: str = "all-MiniLM-L6-v2"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedder = SentenceTransformer(embedder_model)
        logger.info(f"Connected to Neo4j at {uri}")
        logger.info(f"Loaded embedder: {embedder_model}")

    def write_document(self, doc: dict, extracted: Dict[str, Any]):
        """
        Write extracted KG data for a single document to Neo4j.

        Args:
            doc: Original document dict (idx, title, text).
            extracted: Output from KGExtractor — entities, relations, topics.
        """
        entities = extracted.get("entities", [])
        relations = extracted.get("relations", [])
        topics = extracted.get("topics", [])

        if not entities:
            logger.debug(f"[idx={doc.get('idx')}] No entities extracted, skipping")
            return

        with self.driver.session() as session:
            node_ids = {}

            # Write entities with embeddings
            for idx, entity in enumerate(entities):
                label = entity.get("label", "Other")
                props = flatten_properties(entity.get("properties", {}))
                name = props.get("name")
                if not name:
                    continue

                node_id = f"n{idx}"
                node_ids[name] = node_id

                node_text = create_node_text_for_embedding(label, props)
                try:
                    embedding = self.embedder.encode(node_text, show_progress_bar=False).tolist()
                    props["embedding"] = embedding
                except Exception as e:
                    logger.warning(f"Embedding failed for '{name}': {e}")

                props_str = ", ".join(f"{k}: ${k}" for k in props)
                query = f"MERGE ({node_id}:`{label}` {{ {props_str} }})"
                try:
                    session.run(query, props)
                except Exception as e:
                    logger.warning(f"Failed to insert node '{name}': {e}")
                    props_without_embedding = {k: v for k, v in props.items() if k != "embedding"}
                    props_str = ", ".join(f"{k}: ${k}" for k in props_without_embedding)
                    query = f"MERGE ({node_id}:`{label}` {{ {props_str} }})"
                    try:
                        session.run(query, props_without_embedding)
                    except Exception as e2:
                        logger.warning(f"Failed to insert node '{name}' without embedding: {e2}")

            # Write relationships
            for rel in relations:
                start_name = rel.get("start_entity", {}).get("name")
                end_name = rel.get("end_entity", {}).get("name")
                rel_label = rel.get("label")

                if not start_name or not end_name or not rel_label:
                    continue

                if start_name not in node_ids or end_name not in node_ids:
                    continue

                query = (
                    "MATCH (a), (b) "
                    "WHERE a.name = $start_name AND b.name = $end_name "
                    f"MERGE (a)-[r:`{rel_label}`]->(b)"
                )
                try:
                    session.run(query, {"start_name": start_name, "end_name": end_name})
                except Exception as e:
                    logger.warning(f"Failed to create relationship {start_name} --[{rel_label}]--> {end_name}: {e}")

            # Write topics with embeddings and link to entities
            for topic in topics:
                try:
                    topic_text = f"Topic: {topic}. Category: general theme or subject matter."
                    topic_embedding = self.embedder.encode(topic_text, show_progress_bar=False).tolist()
                    session.run(
                        "MERGE (t:Topic {name: $name}) SET t.embedding = $embedding",
                        {"name": topic, "embedding": topic_embedding},
                    )
                    for entity in entities:
                        name = entity.get("properties", {}).get("name")
                        if name:
                            try:
                                session.run(
                                    """
                                    MATCH (a), (t:Topic)
                                    WHERE a.name = $name AND t.name = $topic
                                    MERGE (a)-[:HAS_TOPIC]->(t)
                                    """,
                                    {"name": name, "topic": topic},
                                )
                            except Exception as e:
                                logger.warning(f"Failed to link '{name}' to topic '{topic}': {e}")
                except Exception as e:
                    logger.warning(f"Failed to create topic '{topic}': {e}")

    def close(self):
        self.driver.close()