"""
_sqlcoder_utils.py

Shared utilities for SQLCoder-70B fine-tuning and inference.
"""
from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = -100

SQLCODER_PROMPT_TEMPLATE = """### Task
Generate a SQL query to answer the following question: `{question}`

### Database Schema
{schema}

### SQL Query
"""

def make_sqlcoder_prompt(question: str, schema: str, sql: Optional[str] = None) -> str:
    """Format question and schema into SQLCoder prompt format."""
    prompt = SQLCODER_PROMPT_TEMPLATE.format(
        question=question,
        schema=schema
    )
    if sql:
        prompt += sql
    return prompt

def clean_sql(text: str) -> str:
    """
    Clean and normalize SQL output from model.
    Ported from custom_oracle_llama for consistency.
    """
    s = (text or "").strip()

    if "```" in s:
        # Match ```sql ... ``` or just ``` ... ```
        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", s, re.DOTALL)
        if match:
            s = match.group(1).strip()
        else:
            s = re.sub(r"```(?:sql)?", "", s).strip()
            s = s.strip("`").strip()

    # Find where SQL starts if there's preamble
    m = re.search(r"\b(WITH|SELECT)\b", s, flags=re.IGNORECASE)
    if m:
        s = s[m.start():].strip()

    # Remove trailing semicolon
    semi = s.find(";")
    if semi != -1:
        s = s[:semi].strip()

    return s

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
