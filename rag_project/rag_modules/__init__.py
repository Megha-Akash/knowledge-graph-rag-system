"""
RAG Modules Package
KG-RAG pipeline: retrieval from Neo4j knowledge graph + answer generation.
"""

from .pipeline import RAGPipeline

__all__ = ["RAGPipeline"]