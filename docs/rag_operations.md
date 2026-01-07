# askGuru-SQL: RAG Operations Manual

This document explains how to manage and update the Retrieval-Augmented Generation (RAG) system for the Oracle EBS NL2SQL ensemble.

## 1. Directory Structure

The RAG system is consolidated within `src/api/rag/`:
- `src/api/rag/data/tables/*.json`: Metadata cards for each Oracle table.
- `src/api/rag/data/fewshots.jsonl`: Gold-standard NL-SQL pairs.
- `src/api/rag/index/bm25/`: Lexical indices (automatically generated).
- `src/api/rag/index/vectors/`: Semantic vector indices (automatically generated).
- `src/api/rag/`: Core RAG logic and indexing scripts.

## 2. Initializing / Updating Indices

Whenever you add new table cards or update `fewshots.jsonl`, you **must** rebuild the indices.

**Note on Permissions and Auth**: To avoid permission errors or authentication issues (e.g., expired tokens), ensure you set the necessary environment variables:
```bash
export HF_HOME=/llamaSFT/hf_home
export TRANSFORMERS_CACHE=/llamaSFT/hf_home
export HF_TOKEN="your_valid_huggingface_token_here"
```

### **Step 2.1: BM25 (Lexical) Indexing**
Run this command to build the keyword-based index:
```bash
python3 src/api/rag/build_bm25.py \
  --tables-dir src/api/rag/data/tables \
  --fewshots src/api/rag/data/fewshots.jsonl \
  --out-dir src/api/rag/index/bm25 \
  --hf-home /llamaSFT/hf_home \
  --hf-token $HF_TOKEN
```

### **Step 2.2: Vector (Semantic) Indexing**
Run this command to build the dense vector index:
```bash
# Add RAG package to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/api/rag
python3 src/api/rag/build_vectors.py \
  --tables-dir src/api/rag/data/tables \
  --fewshots src/api/rag/data/fewshots.jsonl \
  --out-dir src/api/rag/index/vectors \
  --model BAAI/bge-en-icl \
  --hf-home /llamaSFT/hf_home \
  --hf-token $HF_TOKEN
```

## 3. Configuration

RAG behavior is controlled via `config/minimal.json` and `.env`.

- **`ENABLE_RAG`**: Set to `true` in `.env` to enable dynamic schema and few-shot discovery.
- **`RAG_CONFIG_PATH`**: Path to the JSON config (default: `config/minimal.json`).

## 4. How it Works

1. **Table Discovery**: The `Planner` uses Hybrid Search (BM25 + BGE Embeddings) to find relevant tables.
2. **Schema Connection**: It uses Breadth-First Search (BFS) to find the shortest path between tables to ensure they are joinable.
3. **Column Selection**: Prioritizes `**[ESSENTIAL]**` columns and Primary Keys.
4. **Few-Shot Retrieval**: Finds the most semantically similar SQL examples from your gold-standard bank.
5. **Evidence Composition**: Injects domain-specific rules based on the detected intent.

## 5. Troubleshooting

- **Missing Tables**: If RAG fails to find a table, ensure the table has a `.json` card in `src/api/rag/data/tables/` and the BM25 index was rebuilt.
