"""
_preprocessing_sqlcoder.py

SQLCoder-specific preprocessing utilities for SFT training.
Supports the SQLCoder-native prompt format.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

from custom_oracle_sqlcoder._sqlcoder_utils import basic_sql_only_check, IGNORE_TOKEN_ID

def make_preprocess_fn_sqlcoder(tokenizer, training_args, is_eval: bool = False):
    """
    Factory function to create SQLCoder preprocessing functions.
    Optimized for the SQLCoder-native format produced by build_oracle_sft_dataset_sqlcoder.py.
    """
    
    def preprocess_sqlcoder(example: Dict[str, Any]) -> Dict[str, Any]:
        conversations = example["conversations"]
        user_content = conversations[0]["content"]
        assistant_content = conversations[1]["content"]
        
        # In SQLCoder-native format, user_content ends with "### SQL Query\n"
        # We just need to concatenate them for training.
        full_text = user_content + assistant_content
        
        if not is_eval:
            if not basic_sql_only_check(assistant_content):
                example_id = example.get("id", "<no-id>")
                logger.warning(f"Non SQL-only target detected for {example_id}")

        enc = tokenizer(
            full_text, 
            truncation=True, 
            max_length=training_args.model_max_length,
            padding=False,
            return_tensors=None
        )
        
        if is_eval:
            # For eval, we only want the prompt encoded, and the label as raw string
            # But the trainer usually expects labels to be tokenized if using DataCollatorForGeneration
            # Actually, standard SFT trainers want labels for loss calculation.
            # If we are doing generation-based eval, we might need a different approach.
            # Keeping it consistent with askGuru trainer expectations:
            prompt_enc = tokenizer(
                user_content,
                truncation=True,
                max_length=training_args.model_max_length,
                padding=False,
                return_tensors=None
            )
            return {
                "input_ids": prompt_enc["input_ids"],
                "attention_mask": prompt_enc["attention_mask"],
                "labels": assistant_content
            }
        
        # Training logic: mask the prompt tokens in labels
        labels = list(enc["input_ids"])
        
        # Find where assistant content starts
        # We can use tokenizer.encode on user_content to find the split point
        prompt_enc = tokenizer(
            user_content,
            truncation=True,
            max_length=training_args.model_max_length,
            padding=False,
            return_tensors=None
        )
        prompt_len = len(prompt_enc["input_ids"])
        
        # Mask everything up to prompt_len
        for i in range(min(prompt_len, len(labels))):
            labels[i] = IGNORE_TOKEN_ID
            
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels
        }

    return preprocess_sqlcoder
