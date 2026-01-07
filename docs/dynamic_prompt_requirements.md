# Dynamic Prompt Generation Requirements (Steps 0-3)

This document outlines the requirements and logic for the dynamic prompt generation process as implemented in the Agentic RAG system. The process is divided into four primary steps: **Indexing**, **OOD Gate Check**, **Planning Phase**, and **Prompt Composition**.

## Embedding Models & Their Roles

The system utilizes two distinct embedding models, each optimized for a specific stage of the RAG pipeline:

### 1. Primary Retrieval Model: `BAAI/bge-en-icl`
- **Role**: Used for **Semantic Search and Vector DB Indexing**.
- **Scope**: Step 0 (Indexing) and Step 2 (Planning/Retrieval).
- **Function**: 
    - Converts table cards and few-shot examples into high-dimensional vectors during the indexing phase.
    - Used during inference to embed the user's question and perform similarity searches against the FAISS indices to retrieve the most relevant tables and few-shot examples.
- **Why**: Chosen for its high performance in information retrieval and in-context learning tasks.

### 2. Similarity Marking Model: `sentence-transformers/all-MiniLM-L6-v2`
- **Role**: Used for **Real-time Semantic Similarity Comparison**.
- **Scope**: Step 3 (Prompt Composition).
- **Function**:
    - Performs a direct "Question vs. Example" similarity check on the few-shot examples selected by the planner.
    - If the cosine similarity exceeds defined thresholds, it triggers the insertion of high-priority markers (`[* EXACT MATCH]` or `[* VERY SIMILAR]`) and specific instructions in the final prompt to "copy the SQL structure exactly."
- **Why**: A lightweight, fast model ideal for real-time comparisons on the API server without significant latency or GPU memory overhead.

---

## Step 0: Indexing (Preparation)

Before the dynamic prompt can be generated, the system must build lexical and semantic indices from the source data (table cards and few-shot examples).

### 0.1 Lexical Indexing (BM25)
- **Script**: `rag/src/build_bm25.py`
- **Objective**: Create BM25 indices for keyword-based retrieval.
- **Indices Built**:
    - **Tables Index** (`tables_bm25.json`): Tokenizes table names, descriptions, column names, and join hints.
    - **Columns Index** (`columns_bm25.json`): Tokenizes individual column metadata for fine-grained retrieval.
    - **Few-shots Index** (`fewshots_bm25.json`): Tokenizes the intents (questions), tables, and SQL queries of the few-shot bank.

### 0.2 Semantic Indexing (Vector DB)
- **Script**: `rag/src/build_vectors.py`
- **Objective**: Create dense vector indices for semantic similarity search using FAISS.
- **Process**:
    - Uses the **`BAAI/bge-en-icl`** embedding model via `EmbedderManager`.
    - Generates embeddings for adapted table cards and few-shot examples.
    - **Indices Built**:
        - `tables.faiss`: FAISS index for table semantic retrieval.
        - `fewshots.faiss`: FAISS index for few-shot semantic retrieval.
    - **Metadata**: Stores mapping IDs (`tables.ids.json`, `fewshots.ids.json`) to link vectors back to original records.

---

## Step 1: Out-of-Domain (OOD) Gate Check

### Objective
To determine if the user's question is within the scope of the available knowledge base (RAG) before proceeding with expensive planning or generation operations.

### Requirements
- **Indexing**: A BM25 index of all available database tables (`tables_bm25.json`) must be present.
- **Scoring**:
    - Perform a BM25 search of the user's question against the table index.
    - Calculate a maximum relevance score.
- **Thresholds**:
    - `t_table_min`: Minimum score threshold (Default: `0.01`).
    - `need_any_hits`: Minimum number of table matches required (Default: `1`).
- **Logic**:
    - If the search returns at least one hit AND the top score is above the threshold, the question is marked **In-Scope**.
    - If either condition fails, the system returns a graceful "out of scope" message: *"unable to answer the question as its not part of the rag."*

---

## Step 2: Planning Phase - Table and Column Selection

### Objective
To identify the most relevant database schema components (tables and columns) and retrieve contextual examples (few-shots) to guide the LLM.

### 2.1 Table Retrieval
- **Keyword Routing**: Use regex-based mapping (`keywords.json`) to identify "seed" tables directly mentioned or implied by the user's question.
- **Hybrid Search**: Combine BM25 (lexical) and Vector (semantic) search results using **Reciprocal Rank Fusion (RRF)** to find the top relevant tables.
- **Selection**: Combine seed tables with hybrid search results, capping the total at a configurable limit (e.g., 8 tables).

### 2.2 Schema Connection
- **Graph Construction**: Build a schema graph where nodes are tables and edges are relationships (Foreign Keys, Join Hints, and External Joins).
- **Shortest Path**: Use Breadth-First Search (BFS) to find the shortest path between disconnected tables to ensure the final schema is joinable.
- **Join Hints**: Collect specific join conditions (e.g., `A.ID = B.A_ID`) to include in the prompt.

### 2.3 Column Selection
- **Prioritization**:
    1. **Essential Columns**: Mandatory columns defined in the table metadata (`[ESSENTIAL]` marker).
    2. **Primary Keys**: Columns designated as PKs.
    3. **Key Columns**: Important columns for filtering or joining.
    4. **Top N Columns**: Remaining columns up to a per-table limit.
- **Schema String**: Build a DDL-like representation of the selected tables and columns, including data types and descriptions.

### 2.4 Few-shot Retrieval
- **Hybrid Search**: Perform RRF-based search against a bank of SQL examples (`fewshots.jsonl`).
- **Selection Strategy**:
    - Retrieve top-ranked semantic matches.
    - Prioritize examples with high table overlap with the current plan.
    - Limit the total count (e.g., 4-6 examples).

---

## Step 3: Prompt Composition

### Objective
To assemble the gathered information into a structured, instruction-heavy prompt for the XiYanSQL-32B (or equivalent) model.

### Components
1. **System Persona**: Explicitly define the model as a "PostgreSQL expert".
2. **User Question**: The original or enriched question from the user.
3. **Database Schema Block**: The DDL-like string generated in Step 2.3.
4. **Reference Information Block**:
    - **Minimal Rules**: Hardcoded SQL best practices and guards.
    - **Similarity Markers**: Special warnings generated using **`all-MiniLM-L6-v2`** if a "Very Similar" example is found (Similarity > 0.80).
    - **Few-Shot Examples**: Formatted Question-Table-SQL triplets retrieved in Step 2.4.
5. **Guidance (Optional)**: Dynamic reasoning or planning tips if available.

### Formatting Requirements
- **PostgreSQL Focus**: All instructions must point towards Postgres-compliant syntax.
- **Essential Column Enforcement**: Explicit instructions to prioritize `**[ESSENTIAL]**` columns.
- **SQL Guardrails**: Rules to avoid common errors (e.g., division by zero, invalid CTE names).
- **Structure**: The prompt follows a fixed template:
    ```text
    [System Instructions]
    [User Question]
    [Database Schema]
    [Reference Information]
    [User Question (Repeated)]
    ```sql
    ```
