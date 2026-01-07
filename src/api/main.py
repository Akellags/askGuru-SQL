import httpx
import asyncio
import time
import json
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from .config import settings
from .schemas import SQLRequest, SQLResponse, ErrorResponse
from .utils import (
    load_mschema, 
    get_filtered_schema, 
    build_llama_prompt, 
    build_sqlcoder_prompt,
    build_critic_prompt,
    get_filtered_schema_from_rag
)
from .db_connector import validate_sql_with_oracle
from .rag import RAGEngine

# Import guardrails
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from custom_oracle_llama.inference.sql_guardrail import clean_sql, is_unsafe
from custom_oracle_sqlcoder.sqlcoder_join_validator import validate_sql_joins

app = FastAPI(title=settings.PROJECT_NAME)

# RAG Engine Initialization
RAG_ENGINE = None
if settings.ENABLE_RAG:
    try:
        with open(settings.RAG_CONFIG_PATH, 'r') as f:
            rag_config = json.load(f)
        RAG_ENGINE = RAGEngine(rag_config)
        print("DONE: RAG Engine initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize RAG Engine: {e}")

# Structured Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("askguru_api")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Extract request body for logging (careful with large bodies)
    body = await request.body()
    try:
        body_json = json.loads(body) if body else {}
    except:
        body_json = {"raw": str(body)}

    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    log_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": round(process_time * 1000, 2),
        "request_body": body_json,
        "client_ip": request.client.host if request.client else "unknown"
    }
    
    logger.info(json.dumps(log_data))
    return response

# Security Setup
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not settings.API_KEY:
        return None # Security disabled if no API_KEY set
    if api_key_header == settings.API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
    )

# Global M-Schema cache
MSCHEMA_CACHE = load_mschema(settings.MSCHEMA_PATH)

async def call_model(url: str, prompt: str) -> str:
    """Helper to call vLLM OpenAI-compatible endpoint."""
    endpoint = f"{url.rstrip('/')}/completions"
    
    print(f"\n{'-'*20} Sending Prompt to vLLM {'-'*20}")
    print(prompt)
    print(f"{'-'*60}\n")

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                endpoint,
                json={
                    "model": settings.MODEL_NAME,
                    "prompt": prompt,
                    "max_tokens": settings.MAX_TOKENS,
                    "temperature": settings.TEMPERATURE,
                    "stop": ["```", ";"]
                }
            )
            if response.status_code != 200:
                print(f"Error from {endpoint}: {response.status_code} - {response.text}")
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["text"].strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model call failed at {endpoint}: {str(e)}")

async def run_fallback_strategy(request: SQLRequest, schema_text: str, rag_context: Optional[dict] = None) -> SQLResponse:
    # 1. Initial Generation (LLaMA)
    llama_prompt = build_llama_prompt(request, MSCHEMA_CACHE, schema_text, rag_context)
    generated_text = await call_model(settings.PRIMARY_MODEL_URL, llama_prompt)
    final_sql = clean_sql(generated_text)
    model_used = "LLaMA-3.1-70B (Fallback)"
    
    # 2. Static Safety Check
    safety_error = is_unsafe(final_sql)
    if safety_error:
        return SQLResponse(
            sql=final_sql,
            model_used=model_used,
            validation_passed=False,
            error=f"Static safety check failed: {safety_error}"
        )

    # 3. Oracle DB Validation (Critic Loop - Round 1)
    is_valid, db_error, _ = validate_sql_with_oracle(final_sql)
    
    if not is_valid:
        # Critic Loop: Ask the model to fix the error once
        critic_prompt = build_critic_prompt(request, MSCHEMA_CACHE, schema_text, final_sql, db_error)
        generated_text = await call_model(settings.PRIMARY_MODEL_URL, critic_prompt)
        final_sql = clean_sql(generated_text)
        
        # Re-validate after correction
        is_valid, db_error, _ = validate_sql_with_oracle(final_sql)

    # 4. Secondary Model Fallback if still invalid
    if not is_valid and settings.ENABLE_SECONDARY_MODEL:
        sqlcoder_prompt = build_sqlcoder_prompt(request, MSCHEMA_CACHE, schema_text)
        generated_text = await call_model(settings.SECONDARY_MODEL_URL, sqlcoder_prompt)
        final_sql = clean_sql(generated_text)
        model_used = "SQLCoder-70B (Fallback)"
        
        # Validate SQLCoder
        is_valid, db_error, _ = validate_sql_with_oracle(final_sql)

    return SQLResponse(
        sql=final_sql,
        model_used=model_used,
        validation_passed=is_valid,
        error=db_error
    )

async def run_voting_strategy(request: SQLRequest, schema_text: str, rag_context: Optional[dict] = None) -> SQLResponse:
    # 1. Parallel Generation
    llama_prompt = build_llama_prompt(request, MSCHEMA_CACHE, schema_text, rag_context)
    sqlcoder_prompt = build_sqlcoder_prompt(request, MSCHEMA_CACHE, schema_text)
    
    tasks = [
        call_model(settings.PRIMARY_MODEL_URL, llama_prompt),
        call_model(settings.SECONDARY_MODEL_URL, sqlcoder_prompt)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    candidates = []
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            continue
        sql = clean_sql(res)
        model_name = "LLaMA-3.1-70B" if idx == 0 else "SQLCoder-70B"
        
        # Validate
        is_valid, db_error, _ = validate_sql_with_oracle(sql)
        join_valid, join_error = validate_sql_joins(sql, schema_text)
        
        # Scoring heuristic
        score = 0
        if is_valid: score += 10
        if join_valid: score += 5
        if not is_unsafe(sql): score += 5
        
        candidates.append({
            "sql": sql,
            "model": model_name,
            "score": score,
            "is_valid": is_valid,
            "error": db_error
        })
    
    if not candidates:
        raise HTTPException(status_code=500, detail="All models failed to generate SQL")
    
    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]
    
    return SQLResponse(
        sql=best["sql"],
        model_used=f"{best['model']} (Voting)",
        validation_passed=best["is_valid"],
        error=best["error"]
    )

@app.post("/generate-sql", 
          response_model=SQLResponse, 
          responses={500: {"model": ErrorResponse}},
          dependencies=[Depends(get_api_key)])
async def generate_sql(request: SQLRequest):
    print(f"\n{'='*20} Incoming Request {'='*20}")
    print(f"Question: {request.question}")
    
    # 1. RAG Context Retrieval (Optional)
    rag_context = None
    schema_text = ""
    
    if settings.ENABLE_RAG and RAG_ENGINE:
        try:
            rag_context = RAG_ENGINE.get_dynamic_context(request.question)
            schema_text = get_filtered_schema_from_rag(rag_context)
            print(f"RAG Tables: {rag_context['tables']}")
        except Exception as e:
            print(f"RAG retrieval failed, falling back to static schema: {e}")
    
    # Fallback to static schema if RAG failed or is disabled
    if not schema_text:
        schema_text = get_filtered_schema(MSCHEMA_CACHE, request.tables)
        print(f"Static Tables: {request.tables}")
    
    if settings.ENSEMBLE_STRATEGY == "voting" and settings.ENABLE_SECONDARY_MODEL:
        return await run_voting_strategy(request, schema_text, rag_context)
    else:
        return await run_fallback_strategy(request, schema_text, rag_context)

@app.get("/health")
async def health():
    return {"status": "ok", "mschema_loaded": len(MSCHEMA_CACHE) > 0}
