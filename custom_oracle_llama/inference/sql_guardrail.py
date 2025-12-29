#!/usr/bin/env python3
"""
sql_guardrail.py

Minimal SQL guardrail + one-retry helper for NL2SQL inference.

Goals:
- Enforce "SQL only" output
- Block unsafe statements (DML/DDL) unless explicitly allowed
- Optionally validate with Oracle parse/EXPLAIN (hook left as optional)

This module is model/serving-engine agnostic. You can call validate_sql() after generation.
If invalid, call build_retry_prompt() to ask the model to regenerate.

Best practice: keep output tokens low (<=512) and use a validator loop.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

UNSAFE_PREFIXES = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "MERGE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "GRANT",
    "REVOKE",
    "BEGIN",
    "DECLARE",
)

SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL)


@dataclass
class ValidationResult:
    """Result of SQL validation."""
    ok: bool
    reason: Optional[str] = None
    cleaned_sql: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


def clean_sql(text: str) -> str:
    """
    Clean and normalize SQL output from model.
    
    Handles:
    - Markdown code fences (```sql ... ```)
    - Explanatory text before/after SQL
    - Semicolons at end
    """
    s = (text or "").strip()

    if "```" in s:
        match = SQL_FENCE_RE.search(s)
        if match:
            s = match.group(1).strip()
        else:
            s = SQL_FENCE_RE.sub("", s).strip()

    m = re.search(r"\b(WITH|SELECT)\b", s, flags=re.IGNORECASE)
    if m:
        s = s[m.start():].strip()

    semi = s.find(";")
    if semi != -1:
        s = s[:semi].strip()

    return s


def is_unsafe(sql: str, allow_dml: bool = False) -> Optional[str]:
    """
    Check if SQL statement is unsafe (DML/DDL).
    
    Returns:
        None if safe, otherwise a reason string like "unsafe_statement:INSERT"
    """
    s = (sql or "").lstrip()
    if not s:
        return "empty_sql"
    
    up = s.upper()
    if up.startswith("WITH") or up.startswith("SELECT"):
        return None
    
    if allow_dml:
        return None
    
    for p in UNSAFE_PREFIXES:
        if up.startswith(p):
            return f"unsafe_statement:{p}"
    
    return "not_select_or_with"


def validate_sql(
    generated_text: str,
    allow_dml: bool = False,
    oracle_parse_hook: Optional[Callable[[str], Tuple[bool, Optional[str], Optional[Dict]]]] = None,
) -> ValidationResult:
    """
    Validate generated SQL.
    
    Args:
        generated_text: Raw text output from model
        allow_dml: If True, allow DML/DDL statements (not recommended)
        oracle_parse_hook: Optional callable(cleaned_sql) -> (ok, error, meta)
            Hook to Oracle EXPLAIN/parse if available
    
    Returns:
        ValidationResult with ok flag and cleaned SQL
    """
    cleaned = clean_sql(generated_text)

    unsafe_reason = is_unsafe(cleaned, allow_dml=allow_dml)
    if unsafe_reason:
        logger.warning(f"Unsafe SQL detected: {unsafe_reason}")
        return ValidationResult(ok=False, reason=unsafe_reason, cleaned_sql=cleaned)

    if oracle_parse_hook is not None:
        try:
            ok, err, meta = oracle_parse_hook(cleaned)
            if not ok:
                logger.warning(f"Oracle parse failed: {err}")
                return ValidationResult(ok=False, reason=f"oracle_parse:{err}", cleaned_sql=cleaned, meta=meta)
            return ValidationResult(ok=True, cleaned_sql=cleaned, meta=meta)
        except Exception as e:
            logger.error(f"Oracle parse exception: {e}")
            return ValidationResult(ok=False, reason=f"oracle_parse_exception:{e}", cleaned_sql=cleaned)

    return ValidationResult(ok=True, cleaned_sql=cleaned)


def build_retry_prompt(
    original_user_prompt: str,
    invalid_sql: str,
    error_reason: str,
) -> str:
    """
    Build retry instruction when validation fails.
    
    Keeps it short and strict to encourage correction.
    
    Args:
        original_user_prompt: Original system/user prompt
        invalid_sql: The SQL that failed validation
        error_reason: Reason for validation failure
    
    Returns:
        Formatted retry prompt for model regeneration
    
    Raises:
        ValueError: If original_user_prompt is empty
    """
    if not original_user_prompt or not original_user_prompt.strip():
        raise ValueError("original_user_prompt cannot be empty")
    
    return (
        original_user_prompt.strip()
        + "\n\n"
        + "The previous SQL was invalid.\n"
        + f"Reason: {error_reason}\n"
        + "Return ONLY a corrected single Oracle SQL statement. No markdown, no explanation.\n"
        + "Previous SQL:\n"
        + invalid_sql.strip()
    )


def example_oracle_parse_hook(cleaned_sql: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Example stub for Oracle parsing hook.
    
    In production, connect to a staging Oracle DB and run:
    - EXPLAIN PLAN FOR <sql>
    Or parse via an Oracle SQL parser if available.
    
    Returns:
        (ok: bool, error: str | None, meta: dict | None)
    """
    return True, None, {"note": "stub - implement Oracle validation"}


class SQLGuardrail:
    """Convenient wrapper for SQL validation and retry logic."""
    
    def __init__(self, allow_dml: bool = False, oracle_parse_hook: Optional[Callable] = None):
        """
        Initialize guardrail.
        
        Args:
            allow_dml: Allow DML/DDL statements
            oracle_parse_hook: Optional hook for Oracle parse validation
        """
        self.allow_dml = allow_dml
        self.oracle_parse_hook = oracle_parse_hook
    
    def validate(self, generated_text: str) -> ValidationResult:
        """Validate SQL output."""
        return validate_sql(
            generated_text,
            allow_dml=self.allow_dml,
            oracle_parse_hook=self.oracle_parse_hook,
        )
    
    def retry_prompt(
        self,
        original_prompt: str,
        invalid_sql: str,
        error_reason: str,
    ) -> str:
        """Build retry prompt."""
        return build_retry_prompt(original_prompt, invalid_sql, error_reason)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_sql = "SELECT * FROM ap_invoices WHERE invoice_amount > 1000"
    result = validate_sql(test_sql)
    print(f"Test 1 - Valid SQL: {result}")
    
    test_sql_bad = "DROP TABLE ap_invoices"
    result = validate_sql(test_sql_bad)
    print(f"Test 2 - Unsafe SQL: {result}")
    
    test_sql_markdown = "```sql\nSELECT * FROM po_headers\n```"
    result = validate_sql(test_sql_markdown)
    print(f"Test 3 - Markdown SQL: {result}")
    
    test_sql_empty = ""
    result = validate_sql(test_sql_empty)
    print(f"Test 4 - Empty SQL: {result}")
