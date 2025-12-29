"""
_preprocessing_utils.py

Shared preprocessing utilities for SFT training scripts.
Eliminates code duplication between LoRA and QLoRA training.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = -100


def basic_sql_only_check(sql: str) -> bool:
    """
    Lightweight guard to catch obviously non-SQL outputs in training data.
    
    Args:
        sql: SQL string to validate
    
    Returns:
        True if SQL appears valid (starts with SELECT or WITH, no markdown)
    """
    if not sql:
        return False
    s = sql.strip()
    s_upper = s.upper()
    
    if not (s_upper.startswith("SELECT") or s_upper.startswith("WITH")):
        return False
    
    if "```" in s:
        return False
    
    return True


def make_preprocess_fn(tokenizer, training_args, is_eval: bool = False):
    """
    Factory function to create preprocessing functions for training/eval datasets.
    
    Args:
        tokenizer: Tokenizer to use for encoding
        training_args: Training configuration (contains model_max_length)
        is_eval: If True, creates eval preprocessing; else creates train preprocessing
    
    Returns:
        Preprocessing function suitable for dataset.map()
    """
    if is_eval:
        def preprocess_eval(example: Dict[str, Any]) -> Dict[str, Any]:
            conversations = example["conversations"]
            prompt = tokenizer.apply_chat_template(
                conversations[:1], 
                tokenize=False, 
                add_generation_prompt=True
            )
            enc = tokenizer(prompt, truncation=True, max_length=training_args.model_max_length)
            label_sql = conversations[1]["content"]
            return {
                "input_ids": enc["input_ids"], 
                "attention_mask": enc["attention_mask"], 
                "labels": label_sql
            }
        return preprocess_eval
    else:
        def preprocess_train(example: Dict[str, Any]) -> Dict[str, Any]:
            conversations = example["conversations"]
            text = tokenizer.apply_chat_template(
                conversations, 
                tokenize=False, 
                add_generation_prompt=False
            )
            enc = tokenizer(text, truncation=True, max_length=training_args.model_max_length)

            target = conversations[1]["content"]
            if not basic_sql_only_check(target):
                example_id = example.get("id", "<no-id>")
                logger.warning(f"Non SQL-only target detected for {example_id}")

            idx = text.find(target)
            target_idx = enc.char_to_token(idx) if idx >= 0 else None
            labels = enc["input_ids"].copy()

            if target_idx is not None:
                labels[:target_idx] = [IGNORE_TOKEN_ID] * target_idx
            else:
                example_id = example.get("id", "<no-id>")
                logger.warning(f"Failed to align target span for {example_id}; training on full sequence.")
            
            return {
                "input_ids": enc["input_ids"], 
                "attention_mask": enc["attention_mask"], 
                "labels": labels
            }
        return preprocess_train
