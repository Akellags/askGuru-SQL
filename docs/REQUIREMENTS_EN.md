# Requirements Document: Oracle EBS NL2SQL Ensemble System (FastAPI + vLLM)

## 1. Project Goal
Develop and deploy a production-grade **Natural Language to SQL (NL2SQL)** ensemble system specialized for **Oracle E-Business Suite (EBS)**. The system will leverage fine-tuned LLaMA-3.1-70B and SQLCoder-70B models to provide high-accuracy SQL generation, served via FastAPI and optimized with vLLM.

---

## 2. Architecture Overview

### 2.1 Model Ensemble Strategy
- **Primary Model**: Fine-tuned **LLaMA-3.1-70B-Instruct** (merged & 4-bit quantized). Specialized in complex reasoning and multi-table joins.
- **Secondary Model**: Fine-tuned **SQLCoder-70B** (merged & 4-bit quantized). Specialized in pure SQL generation and simple queries.
- **Ensemble Control**:
  - **ENABLE_SECONDARY_MODEL**: A boolean flag (environment variable) to enable or disable the secondary model. When disabled, only the primary model is used.
- **Routing Logic**:
  - **Sequential Fallback**: Primary model is called first. If validation fails or confidence is low, the secondary model is triggered.
  - **Ensemble Voting**: Both models are called in parallel, and a scoring heuristic (based on JOIN validation) selects the best candidate.

### 2.2 Inference Engine
- **vLLM**: Used for high-throughput, low-latency serving of 70B models on A100-80GB GPUs.
- **Configuration**:
  - `max_model_len`: 8192 - 16384 tokens.
  - `max_num_seqs`: 4 (target concurrency).
  - `quantization`: AWQ (4-bit).

### 2.3 Orchestration Layer (FastAPI)
- **Endpoint**: `/generate-sql`
- **Input**: User question, optional context.
- **Internal Logic**:
  1. Load static M-Schema from `askGuru_m-schema_converted.txt`.
  2. Assemble prompt using model-specific templates.
  3. Call vLLM endpoint(s).
  4. Post-process & Validate SQL.
  5. (Optional) Execute on Oracle DB for final verification.

---

## 3. Data & Schema Requirements

### 3.1 M-Schema Integration
- Use the **Static M-Schema** format for prompt construction.
- The system must be able to load schema metadata from `data/data_warehouse/oracle_train/askGuru_m-schema_converted.txt` instead of querying the database for every request.

### 3.2 Prompt Templates
- **LLaMA Template**: Chat-based format with `【database schema】` and `[User Question]` headers.
- **SQLCoder Template**: Structured format with `### Task`, `### Database Schema`, and `### SQL Query`.

---

## 4. Validation & Safety

### 4.1 SQL Guardrails
- **Clean SQL**: Strip markdown fences and explanatory text.
- **Unsafe Statement Blocking**: Prevent `DML` (INSERT, UPDATE, DELETE) and `DDL` (DROP, ALTER) operations.
- **Join Validation**: Specifically validate table aliases and join conditions (crucial for SQLCoder).

### 4.2 Oracle DB Verification (Thin Client)
- **Connection**: Use **Oracle Thin Client** for lightweight connectivity.
- **Critic Loop**: If enabled, execute the generated SQL on a staging Oracle instance. If an error occurs, capture the traceback and pass it back to the model for a self-correction attempt.

---

## 5. Infrastructure & Performance

### 5.1 Hardware
- **Inference**: 1× NVIDIA A100-80GB (per 70B model).
- **Network**: Low-latency connectivity between FastAPI and vLLM servers.

### 5.2 Targets
- **Concurrency**: 4 concurrent users at peak.
- **Accuracy**: >70% execution match on Oracle EBS benchmark.
- **Latency**: < 5 seconds for end-to-end generation (including ensemble logic).

---

## 6. Project Conventions
- **Code**: All implementations in `src/`.
- **Testing**: All test scripts in `tests/`.
- **DevOps**: All deployment scripts (vLLM, Docker) in `devOps/`.
- **Documentation**: All documentation in `docs/`.
- **Database**: All schema-related files in `dbSchema/`.
