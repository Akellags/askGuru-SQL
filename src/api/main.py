import httpx
import asyncio
from fastapi import FastAPI, HTTPException
from .config import settings
from .schemas import SQLRequest, SQLResponse, ErrorResponse
from .utils import (
    load_mschema, 
    get_filtered_schema, 
    build_llama_prompt, 
    build_sqlcoder_prompt,
    build_critic_prompt
)
from .db_connector import validate_sql_with_oracle

# Import guardrails
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from custom_oracle_llama.inference.sql_guardrail import clean_sql, is_unsafe
from custom_oracle_sqlcoder.sqlcoder_join_validator import validate_sql_joins

app = FastAPI(title=settings.PROJECT_NAME)

# Global M-Schema cache
MSCHEMA_CACHE = load_mschema(settings.MSCHEMA_PATH)

async def call_model(url: str, prompt: str) -> str:
    """Helper to call vLLM OpenAI-compatible endpoint."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{url}/completions",
                json={
                    "model": "merged_model",
                    "prompt": prompt,
                    "max_tokens": settings.MAX_TOKENS,
                    "temperature": settings.TEMPERATURE,
                    "stop": ["```", ";"]
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["text"].strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model call failed: {str(e)}")

async def run_fallback_strategy(request: SQLRequest, schema_text: str) -> SQLResponse:
    # 1. Initial Generation (LLaMA)
    llama_prompt = build_llama_prompt(request.question, schema_text)
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
        critic_prompt = build_critic_prompt(request.question, schema_text, final_sql, db_error)
        generated_text = await call_model(settings.PRIMARY_MODEL_URL, critic_prompt)
        final_sql = clean_sql(generated_text)
        
        # Re-validate after correction
        is_valid, db_error, _ = validate_sql_with_oracle(final_sql)

    # 4. Secondary Model Fallback if still invalid
    if not is_valid and settings.ENABLE_SECONDARY_MODEL:
        sqlcoder_prompt = build_sqlcoder_prompt(request.question, schema_text)
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

async def run_voting_strategy(request: SQLRequest, schema_text: str) -> SQLResponse:
    # 1. Parallel Generation
    llama_prompt = build_llama_prompt(request.question, schema_text)
    sqlcoder_prompt = build_sqlcoder_prompt(request.question, schema_text)
    
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

@app.post("/generate-sql", response_model=SQLResponse, responses={500: {"model": ErrorResponse}})
async def generate_sql(request: SQLRequest):
    # Prepare Schema
    schema_text = get_filtered_schema(MSCHEMA_CACHE, request.tables)
    
    if settings.ENSEMBLE_STRATEGY == "voting" and settings.ENABLE_SECONDARY_MODEL:
        return await run_voting_strategy(request, schema_text)
    else:
        return await run_fallback_strategy(request, schema_text)

@app.get("/health")
async def health():
    return {"status": "ok", "mschema_loaded": len(MSCHEMA_CACHE) > 0}
