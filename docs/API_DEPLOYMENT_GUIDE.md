# API Orchestration Layer Deployment Guide

This guide covers the deployment of the FastAPI orchestration layer that manages the ensemble of LLaMA and SQLCoder models for Oracle EBS NL2SQL.

## 1. Prerequisites

- Python 3.12+
- Oracle Instant Client (if using Thick mode, otherwise Thin mode is used by default)
- Access to vLLM endpoints for Primary (LLaMA) and Secondary (SQLCoder) models.

## 2. Package Installation

The API requires the following packages. You can install them in your virtual environment:

```bash
sudo apt-get update && sudo apt-get install -y libaio1 libaio-dev libstdc++6
pip install fastapi uvicorn httpx oracledb pydantic-settings python-dotenv
```



## 3. Environment Configuration

Create a `.env` file in the `src/api` directory or set the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PRIMARY_MODEL_URL` | vLLM endpoint for LLaMA-3.1-70B | `http://localhost:8000/v1` |
| `SECONDARY_MODEL_URL` | vLLM endpoint for SQLCoder-70B | `http://localhost:8001/v1` |
| `ENABLE_SECONDARY_MODEL` | Enable fallback/voting to secondary model | `false` |
| `ENSEMBLE_STRATEGY` | `fallback` or `voting` | `fallback` |
| `DATABASE_USER` | Oracle DB Username | `apps` |
| `DATABASE_PASSWORD` | Oracle DB Password | `apps` |
| `DATABASE_DSN` | Oracle DB DSN | `183.82.4.173:1529/VIS` |
| `ORACLE_CLIENT_LIB_UBUNTU` | Path to Oracle Instant Client (Optional) | `""` |
| `API_KEY` | Secret key for API access | `""` |
| `MSCHEMA_PATH` | Path to `askGuru_m-schema_converted.txt` | `data/data_warehouse/oracle_train/askGuru_m-schema_converted.txt` |

## 4. Starting the API Server

Run the following command from the project root:

```bash
# Ensure you are in the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 5. Starting vLLM Servers (Reference)

### Primary Model (LLaMA-3.1-70B)
```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -m vllm.entrypoints.openai.api_server \
  --model /llamaSFT/outputs/merged_oracle_llama70b_awq4 \
  --dtype float16 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  --quantization awq \
  --host 0.0.0.0 \
  --port 8000
```

### Secondary Model (SQLCoder-70B)
```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/sqlcoder-70b-awq \
  --dtype float16 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  --quantization awq \
  --host 0.0.0.0 \
  --port 8001
```

## 6. Testing with CURL

Once the API is running, you can test it with the following CURL command (replace `your-secret-api-key-here` with the value from your `.env`):

```bash
curl -X POST http://localhost:8000/generate-sql \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-secret-api-key-here" \
     -d '{
           "question": "Show me all invoices for vendor ABC created in the last 30 days",
           "tables": ["AP_INVOICES_ALL", "AP_SUPPLIERS"]
         }'
```

### Expected Response Format
```json
{
  "sql": "SELECT ... FROM AP_INVOICES_ALL ...",
  "model_used": "LLaMA-3.1-70B (Fallback)",
  "validation_passed": true,
  "error": null
}
```
