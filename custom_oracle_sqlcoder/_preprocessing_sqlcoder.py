"""
_preprocessing_sqlcoder.py

SQLCoder-specific preprocessing utilities for SFT training.
Converts askGuru conversation format to SQLCoder prompt format.

SQLCoder Format:
### Task
Generate a SQL query to answer the following question: `{question}`

### Database Schema
{schema}

### SQL Query
{sql}
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = -100


def basic_sql_only_check(sql: str) -> bool:
    """Lightweight guard to catch obviously non-SQL outputs."""
    if not sql:
        return False
    s = sql.strip()
    s_upper = s.upper()
    
    if not (s_upper.startswith("SELECT") or s_upper.startswith("WITH")):
        return False
    
    if "```" in s:
        return False
    
    return True


def extract_question_and_schema(user_content: str) -> tuple:
    """
    Extract question and schema from askGuru user prompt.
    """
    lines = user_content.split('\n')
    
    schema_lines = []
    question_lines = []
    in_schema = False
    in_question = False
    
    for line in lines:
        if '[User Question]' in line:
            in_question = True
            in_schema = False
            continue
        
        if in_question:
            if line.strip():
                question_lines.append(line)
        elif line.startswith('# ') or line.startswith('- '):
            in_schema = True
            schema_lines.append(line)
        elif in_schema:
            schema_lines.append(line)
    
    question = ' '.join(question_lines).strip()
    schema = '\n'.join(schema_lines).strip()
    
    return question, schema


def make_sqlcoder_prompt(question: str, schema: str, sql: str = None) -> str:
    """Format question and schema into SQLCoder prompt format."""
    prompt = f"""### Task
Generate a SQL query to answer the following question: `{question}`

### Database Schema
{schema}

### SQL Query
"""
    
    if sql:
        prompt += sql
    
    return prompt


def make_preprocess_fn_sqlcoder(tokenizer, training_args, is_eval: bool = False):
    """
    Factory function to create SQLCoder preprocessing functions.
    """
    
    if is_eval:
        def preprocess_eval_sqlcoder(example: Dict[str, Any]) -> Dict[str, Any]:
            conversations = example["conversations"]
            user_content = conversations[0]["content"]
            assistant_content = conversations[1]["content"]
            
            question, schema = extract_question_and_schema(user_content)
            prompt = make_sqlcoder_prompt(question, schema, sql=None)
            
            enc = tokenizer(prompt, truncation=True, max_length=training_args.model_max_length)
            
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": assistant_content
            }
        return preprocess_eval_sqlcoder
    
    else:
        def preprocess_train_sqlcoder(example: Dict[str, Any]) -> Dict[str, Any]:
            conversations = example["conversations"]
            user_content = conversations[0]["content"]
            sql_content = conversations[1]["content"]
            
            if not basic_sql_only_check(sql_content):
                example_id = example.get("id", "<no-id>")
                logger.warning(f"Non SQL-only target detected for {example_id}")
            
            question, schema = extract_question_and_schema(user_content)
            prompt = make_sqlcoder_prompt(question, schema, sql=sql_content)
            
            enc = tokenizer(prompt, truncation=True, max_length=training_args.model_max_length)
            
            sql_start = prompt.find(sql_content)
            sql_start_idx = enc.char_to_token(sql_start) if sql_start >= 0 else None
            
            labels = enc["input_ids"].copy()
            
            if sql_start_idx is not None:
                labels[:sql_start_idx] = [IGNORE_TOKEN_ID] * sql_start_idx
            else:
                example_id = example.get("id", "<no-id>")
                logger.warning(f"Failed to align SQL span for {example_id}")
            
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels
            }
        return preprocess_train_sqlcoder
