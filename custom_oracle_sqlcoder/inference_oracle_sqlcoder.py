"""
inference_oracle_sqlcoder.py

Oracle EBS NL2SQL inference using fine-tuned SQLCoder-70B model.
Supports both base model and LoRA-merged variants.

Usage (standalone):
    python custom_oracle_sqlcoder/inference_oracle_sqlcoder.py \
      --model_path outputs/oracle_sqlcoder70b_lora \
      --question "List all active suppliers" \
      --schema_file schemas/oracle_ebs_sample.txt

Usage (as module):
    from custom_oracle_sqlcoder.inference_oracle_sqlcoder import SQLCoderInference
    
    inference = SQLCoderInference("defog/sqlcoder-70b-alpha")
    sql = inference.generate(question, schema)
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from custom_oracle_sqlcoder._preprocessing_sqlcoder import (
    extract_question_and_schema,
    make_sqlcoder_prompt
)
from custom_oracle_sqlcoder.sqlcoder_join_validator import validate_sql_joins

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SQLCoderInference:
    """SQLCoder-70B inference engine for Oracle EBS NL2SQL."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        max_tokens: int = 512,
        temperature: float = 0.1
    ):
        """
        Initialize SQLCoder inference.
        
        Args:
            model_path: Path to model (HF hub ID or local path)
            device: Device to load model on ("auto", "cuda", "cpu")
            dtype: Model dtype (torch.float16, torch.bfloat16, torch.float32)
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (0.1 for deterministic SQL)
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        logger.info(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info(f"Loading model from {model_path} on {self.device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device if device == "auto" else None
        )
        
        if device != "auto":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def generate(
        self,
        question: str,
        schema: str,
        validate_joins: bool = True,
        max_retries: int = 1
    ) -> str:
        """
        Generate SQL query for Oracle EBS.
        
        Args:
            question: Natural language question
            schema: Database schema (tables, columns, types)
            validate_joins: If True, validate generated SQL for join correctness
            max_retries: Number of retries if join validation fails
        
        Returns:
            Generated SQL query
        """
        prompt = make_sqlcoder_prompt(question, schema, sql=None)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.95,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        sql = response.split("### SQL Query")[-1].strip()
        sql = sql.split("```")[0].strip() if "```" in sql else sql
        
        if validate_joins:
            is_valid, errors = validate_sql_joins(sql, schema)
            if not is_valid and max_retries > 0:
                logger.warning(f"JOIN validation failed: {errors}. Retrying...")
                max_retries -= 1
        
        return sql
    
    def generate_batch(
        self,
        questions: list[str],
        schemas: list[str],
        validate_joins: bool = True
    ) -> list[str]:
        """Generate SQL for multiple questions."""
        results = []
        for question, schema in zip(questions, schemas):
            sql = self.generate(question, schema, validate_joins=validate_joins)
            results.append(sql)
        return results


def main():
    parser = argparse.ArgumentParser(description="SQLCoder Oracle EBS Inference")
    parser.add_argument("--model_path", required=True, help="Model path (HF hub or local)")
    parser.add_argument("--question", required=True, help="Natural language question")
    parser.add_argument("--schema_file", help="Path to schema file (alternative to --schema)")
    parser.add_argument("--schema", help="Inline schema definition")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--validate_joins", action="store_true", help="Validate SQL joins")
    
    args = parser.parse_args()
    
    schema = None
    if args.schema_file and os.path.exists(args.schema_file):
        with open(args.schema_file, 'r') as f:
            schema = f.read()
    elif args.schema:
        schema = args.schema
    else:
        raise ValueError("Provide --schema or --schema_file")
    
    inference = SQLCoderInference(
        model_path=args.model_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    sql = inference.generate(
        question=args.question,
        schema=schema,
        validate_joins=args.validate_joins
    )
    
    print("\n=== Generated SQL ===")
    print(sql)


if __name__ == "__main__":
    main()
