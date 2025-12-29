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

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


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


SQL_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


@dataclass
class ValidationResult:
    ok: bool
    reason: Optional[str] = None
    cleaned_sql: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


def clean_sql(text: str) -> str:
    s = (text or "").strip()

    # Remove markdown fences if present
    if "```" in s:
        s = SQL_FENCE_RE.sub("", s).strip()

    # If the model accidentally prepends explanations, attempt to extract the first SQL statement.
    # Heuristic: find first occurrence of WITH/SELECT.
    m = re.search(r"\b(WITH|SELECT)\b", s, flags=re.IGNORECASE)
    if m:
        s = s[m.start():].strip()

    # Strip trailing commentary after a second statement marker if present.
    # Keep first statement; Oracle SQL may contain semicolons rarely; if present, keep up to first semicolon.
    semi = s.find(";")
    if semi != -1:
        s = s[:semi].strip()

    return s


def is_unsafe(sql: str, allow_dml: bool = False) -> Optional[str]:
    s = (sql or "").lstrip()
    if not s:
        return "empty_sql"
    up = s.upper()
    # Allow WITH/SELECT
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
    oracle_parse_hook=None,
) -> ValidationResult:
    """
    oracle_parse_hook: optional callable(cleaned_sql) -> (ok:bool, error:str|None, meta:dict|None)
    """
    cleaned = clean_sql(generated_text)

    unsafe_reason = is_unsafe(cleaned, allow_dml=allow_dml)
    if unsafe_reason:
        return ValidationResult(ok=False, reason=unsafe_reason, cleaned_sql=cleaned)

    if oracle_parse_hook is not None:
        try:
            ok, err, meta = oracle_parse_hook(cleaned)
            if not ok:
                return ValidationResult(ok=False, reason=f"oracle_parse:{err}", cleaned_sql=cleaned, meta=meta)
            return ValidationResult(ok=True, cleaned_sql=cleaned, meta=meta)
        except Exception as e:
            return ValidationResult(ok=False, reason=f"oracle_parse_exception:{e}", cleaned_sql=cleaned)

    return ValidationResult(ok=True, cleaned_sql=cleaned)


def build_retry_prompt(
    original_user_prompt: str,
    invalid_sql: str,
    error_reason: str,
) -> str:
    """
    Build a retry instruction. Keep it short and strict.
    """
    return (
        original_user_prompt.strip()
        + "\n\n"
        + "The previous SQL was invalid.\n"
        + f"Reason: {error_reason}\n"
        + "Return ONLY a corrected single Oracle SQL statement. No markdown, no explanation.\n"
        + "Previous SQL:\n"
        + invalid_sql.strip()
    )


# Optional: example oracle parse hook skeleton (developer to implement)
def example_oracle_parse_hook(cleaned_sql: str):
    """
    Example stub.
    In production, connect to a staging Oracle DB and run:
    - EXPLAIN PLAN FOR <sql>
    Or parse via an Oracle SQL parser if available.
    """
    return True, None, {"note": "stub"}
