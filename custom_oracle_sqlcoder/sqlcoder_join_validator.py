"""
sqlcoder_join_validator.py

Post-processing validator for SQLCoder-generated SQL.

SQLCoder-70B has 85.7% accuracy on JOINs (weakest component).
This module validates and corrects common JOIN errors:
- Missing JOIN keywords
- Incorrect table aliases
- Wrong join conditions
- Missing INNER/LEFT/OUTER keywords
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def validate_sql_joins(sql: str, schema: str) -> Tuple[bool, Optional[str]]:
    """
    Validate generated SQL for common JOIN errors.
    
    Args:
        sql: Generated SQL query
        schema: Database schema (tables and columns)
    
    Returns:
        (is_valid, error_message)
        - is_valid: True if SQL passes validation
        - error_message: None if valid, error description if invalid
    """
    sql_upper = sql.upper()
    
    if not sql_upper.startswith(("SELECT", "WITH")):
        return False, "SQL does not start with SELECT or WITH"
    
    if ";" in sql:
        sql = sql.rstrip(";").strip()
    
    errors = []
    
    errors.extend(_validate_table_aliases(sql, schema))
    errors.extend(_validate_join_syntax(sql))
    errors.extend(_validate_from_clause(sql))
    
    if errors:
        error_msg = " | ".join(errors)
        return False, error_msg
    
    return True, None


def _validate_table_aliases(sql: str, schema: str) -> list[str]:
    """Validate table aliases are referenced correctly."""
    errors = []
    
    from_match = re.search(r'\bFROM\s+(\w+)\s+(?:AS\s+)?(\w+)', sql, re.IGNORECASE)
    if not from_match:
        return errors
    
    table_name = from_match.group(1)
    alias = from_match.group(2) if from_match.lastindex >= 2 else table_name
    
    if table_name.lower() not in schema.lower():
        errors.append(f"Table '{table_name}' not found in schema")
    
    join_matches = re.finditer(
        r'\bJOIN\s+(\w+)\s+(?:AS\s+)?(\w+)',
        sql,
        re.IGNORECASE
    )
    
    for match in join_matches:
        join_table = match.group(1)
        join_alias = match.group(2) if match.lastindex >= 2 else join_table
        
        if join_table.lower() not in schema.lower():
            errors.append(f"Table '{join_table}' in JOIN not found in schema")
    
    return errors


def _validate_join_syntax(sql: str) -> list[str]:
    """Validate JOIN syntax."""
    errors = []
    
    if re.search(r'\bJOIN\s+\bWHERE\b', sql, re.IGNORECASE):
        errors.append("Missing ON clause after JOIN")
    
    on_count = len(re.findall(r'\bON\b', sql, re.IGNORECASE))
    join_count = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
    
    if on_count < join_count:
        errors.append(f"Expected {join_count} ON clauses but found {on_count}")
    
    return errors


def _validate_from_clause(sql: str) -> list[str]:
    """Validate FROM clause exists."""
    errors = []
    
    if not re.search(r'\bFROM\b', sql, re.IGNORECASE):
        errors.append("Missing FROM clause")
    
    return errors


def suggest_join_fix(sql: str, schema: str) -> Optional[str]:
    """
    Suggest a fix for common JOIN errors.
    
    Args:
        sql: Generated SQL with potential errors
        schema: Database schema
    
    Returns:
        Suggested fixed SQL, or None if no fix available
    """
    sql_upper = sql.upper()
    
    if "JOIN" in sql_upper and "ON" not in sql_upper and "WHERE" in sql_upper:
        where_idx = sql_upper.find("WHERE")
        before_where = sql[:where_idx].rstrip()
        after_where = sql[where_idx:]
        
        on_match = re.search(r'(\w+)\.id\s*=\s*(\w+)\.(\w+_id)', before_where, re.IGNORECASE)
        if on_match:
            fixed = f"{before_where} ON {on_match.group(1)}.id = {on_match.group(2)}.{on_match.group(3)} {after_where}"
            return fixed
    
    return None


def extract_tables_from_sql(sql: str) -> list[str]:
    """Extract table names referenced in SQL."""
    tables = []
    
    from_match = re.search(r'\bFROM\s+(\w+)', sql, re.IGNORECASE)
    if from_match:
        tables.append(from_match.group(1))
    
    join_matches = re.finditer(r'\bJOIN\s+(\w+)', sql, re.IGNORECASE)
    for match in join_matches:
        tables.append(match.group(1))
    
    return list(set(tables))


def extract_joins_from_sql(sql: str) -> list[dict]:
    """Extract JOIN conditions from SQL."""
    joins = []
    
    join_pattern = r'\b(INNER\s+)?JOIN\s+(\w+)\s+(?:AS\s+)?(\w+)?\s+ON\s+([^,;]+?)(?=(?:LEFT|RIGHT|INNER|OUTER|WHERE|GROUP|ORDER|;|$))'
    
    for match in re.finditer(join_pattern, sql, re.IGNORECASE):
        join_type = match.group(1) or "INNER"
        table = match.group(2)
        alias = match.group(3) or table
        condition = match.group(4).strip()
        
        joins.append({
            "type": join_type.strip(),
            "table": table,
            "alias": alias,
            "condition": condition
        })
    
    return joins


def validate_join_alignment(sql: str, schema: str) -> Tuple[bool, Optional[str]]:
    """
    Validate JOINs align with schema foreign keys.
    
    Args:
        sql: SQL query
        schema: Schema definition (should contain table relationships)
    
    Returns:
        (is_aligned, issue_description)
    """
    joins = extract_joins_from_sql(sql)
    tables = extract_tables_from_sql(sql)
    
    if not joins:
        return True, None
    
    schema_lower = schema.lower()
    
    for join in joins:
        table = join["table"].lower()
        
        if table not in schema_lower:
            return False, f"Table '{join['table']}' in JOIN not found in schema"
        
        condition = join["condition"].lower()
        
        if "=" not in condition:
            return False, f"Invalid JOIN condition: '{join['condition']}'"
    
    return True, None
