# KG-RAG: Knowledge Graph Retrieval-Augmented Generation

An end-to-end pipeline that builds a knowledge graph from text corpora and uses it for grounded question answering.

## Overview

Most RAG systems retrieve flat text chunks. This system instead builds a **structured knowledge graph** (Neo4j) from the corpus, then retrieves relevant subgraphs using vector similarity and multi-hop traversal — giving the LLM structured, relational context rather than raw passages.

```
Corpus (HotpotQA)
      ↓
 KG Builder — LLM extracts entities, relations, topics → Neo4j
      ↓
 KG-RAG Pipeline
      ├── Vector similarity → seed entity finding
      ├── Multi-hop graph traversal → subgraph retrieval
      ├── Cosine similarity reranking → top-k context paths
      └── Answer generation → grounded answer
```

## Features

- **LLM-based KG extraction** — entities, relations, and topics extracted from raw text using a prompted LLM (Ollama)
- **Embedding-aware Neo4j writes** — every node stored with a sentence-transformer embedding for vector retrieval
- **Vector seed entity finding** — question is embedded and matched against node embeddings via cosine similarity, replacing brittle regex-based entity linking
- **Multi-hop graph traversal** — variable-length path queries retrieve multi-hop relational context
- **Cosine similarity reranking** — retrieved paths reranked by semantic similarity to the question
- **Resumable KG building** — checkpoint system allows resuming interrupted corpus builds
- **Interactive + batch modes** — ask single questions or process full corpora via CLI

## Project Structure

```
rag/
├── kg_builder/              # KG construction pipeline
│   ├── __init__.py
│   ├── dataset_loader.py    # HotpotQA / MuSiQue corpus loader
│   ├── extractor.py         # LLM-based entity/relation extraction
│   ├── prompt_templates.py  # Extraction prompts
│   ├── neo4j_writer.py      # Neo4j writes with embeddings
│   ├── checkpointer.py      # Resumable build checkpointing
│   └── builder_main.py      # Orchestration
├── rag_modules/             # RAG pipeline
│   ├── __init__.py
│   ├── retriever.py         # Vector search + multi-hop traversal
│   ├── generator.py         # Answer generation via Ollama
│   └── pipeline.py          # End-to-end orchestration
├── scripts/
│   ├── build_kg.py          # CLI: build knowledge graph
│   └── run_rag.py           # CLI: query the RAG pipeline
└── configs/
    └── pipeline_config.example.yaml
```

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Neo4j

Start a Neo4j instance and update your config with the URI and credentials.

### Ollama

Serve a model via Ollama:

```bash
ollama run gemma3:27b
```

### Config

Copy the example config and fill in your values:

```bash
cp configs/pipeline_config.example.yaml configs/pipeline_config.yaml
```

Edit `pipeline_config.yaml`:

```yaml
neo4j:
  uri: bolt://YOUR_HOST:7687
  user: neo4j
  password: YOUR_PASSWORD

llm:
  model: gemma3:27b
  endpoint: http://YOUR_OLLAMA_HOST:11434/api/generate

embedder:
  model: all-MiniLM-L6-v2

data:
  corpus_path: path/to/hotpotqa_corpus.json
```

## Usage

### Build the Knowledge Graph

```bash
# Test on 10 documents
python scripts/build_kg.py --max-docs 10

# Full build (resumable)
python scripts/build_kg.py

# Resume interrupted build
python scripts/build_kg.py --resume

# Rebuild from scratch
python scripts/build_kg.py --reset

# Build from a single text (quick test)
python scripts/build_kg.py --text "Christopher Nolan directed Inception in 2010."
```

### Query the Pipeline

```bash
# Single question
python scripts/run_rag.py --question "Who created Sherlock Holmes?"

# With verbose step-by-step output
python scripts/run_rag.py --question "Who created Sherlock Holmes?" --verbose

# Interactive mode
python scripts/run_rag.py --interactive
```

### Example Output

```
=== KG-RAG Pipeline ===
Question: Who created Sherlock Holmes?

[1] Retrieving context from knowledge graph...
    Seed entities: ['Sherlock Holmes', 'The Exploits of Sherlock Holmes', 'Sir Arthur Conan Doyle']
    Retrieved 15 context paths
    Top 3 context paths:
      - Sir Arthur Conan Doyle --[CREATED]--> Sherlock Holmes
      - Sherlock Holmes --[FEATURED_IN]--> The Exploits of Sherlock Holmes
      - Adrian Conan Doyle --[WROTE]--> The Adventure of the Seven Clocks

[2] Generating answer...

=== Answer ===
Sir Arthur Conan Doyle.
```

## Dataset

Tested on the [HotpotQA](https://hotpotqa.github.io/) corpus — a multi-hop question answering dataset requiring reasoning over multiple supporting documents. Also compatible with MuSiQue and 2WikiMultiHopQA corpora using the same format.

## Tech Stack

- **Neo4j** — graph database for KG storage and traversal
- **Ollama** — local LLM serving for extraction and generation
- **sentence-transformers** — node and query embeddings (`all-MiniLM-L6-v2`)
- **Python** — neo4j driver, requests, PyYAML