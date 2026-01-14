#!/usr/bin/env python3
"""
build_oracle_sft_dataset_sqlcoder.py

Oracle EBS NL2SQL SFT Dataset Pipeline for SQLCoder:
- Custom prompt format optimized for Defog SQLCoder-70B
- Preserves all augmentation and validation logic from Llama version
- Outputs separate dataset to avoid mixing structures
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import yaml
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from custom_oracle_sqlcoder._sqlcoder_utils import SQLCODER_PROMPT_TEMPLATE, make_sqlcoder_prompt

# ============================================================================
# PHASE 2: DATA EXTRACTION & TRANSFORMATION
# ============================================================================

def load_main_training_data(filepath: str) -> List[Dict[str, Any]]:
    """Load main training data JSONL"""
    examples = []
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        logger.info(f"✓ Loaded {len(examples)} examples from main training data")
        return examples
    except Exception as e:
        logger.error(f"Failed to load main training data: {e}")
        return []

def load_corrections(filepath: str) -> List[Dict[str, Any]]:
    """Load corrections.jsonl"""
    examples = []
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        logger.info(f"✓ Loaded {len(examples)} high-quality corrections")
        return examples
    except Exception as e:
        logger.error(f"Failed to load corrections: {e}")
        return []

def load_joins(filepath: str) -> List[str]:
    """Load joins.json"""
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            joins = json.load(f)
        logger.info(f"✓ Loaded {len(joins)} join paths")
        return joins if isinstance(joins, list) else []
    except Exception as e:
        logger.error(f"Failed to load joins: {e}")
        return []

def load_table_schemas(folder: str) -> Dict[str, Dict[str, Any]]:
    """Load all *.json table schema files"""
    schemas = {}
    if not os.path.exists(folder):
        logger.warning(f"Folder not found: {folder}")
        return {}
    try:
        for filename in os.listdir(folder):
            if filename.endswith(".json") and not filename.startswith("joins"):
                filepath = os.path.join(folder, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        schema = json.load(f)
                    table_name = filename.replace(".json", "")
                    schemas[table_name] = schema
                except Exception as e:
                    logger.warning(f"Failed to load schema {filename}: {e}")
        logger.info(f"✓ Loaded {len(schemas)} table schemas")
        return schemas
    except Exception as e:
        logger.error(f"Failed to load table schemas: {e}")
        return {}

def load_keywords(filepath: str) -> Dict[str, str]:
    """Load keywords.json"""
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            keywords = json.load(f)
        logger.info(f"✓ Loaded {len(keywords)} keyword mappings")
        return keywords if isinstance(keywords, dict) else {}
    except Exception as e:
        logger.error(f"Failed to load keywords: {e}")
        return {}

# ============================================================================
# PHASE 2: DATA TRANSFORMATION
# ============================================================================

def normalize_sql(sql: str) -> str:
    """Normalize SQL formatting"""
    sql = (sql or "").strip()
    if sql.startswith("```"):
        sql = sql.strip("`").strip()
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    return sql

def extract_tables_from_sql(sql: str) -> List[str]:
    """Extract table names from SQL"""
    if not sql:
        return []
    pattern = r"\b(?:FROM|JOIN|INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN)\s+(\w+)"
    matches = re.findall(pattern, sql, re.IGNORECASE)
    return list(set(m.upper() for m in matches))

def deduplicate_examples(
    examples: List[Dict[str, Any]], 
    key_fields: List[str] = ["question", "sql_gold"]
) -> Tuple[List[Dict[str, Any]], int]:
    """Remove exact duplicates"""
    seen: Set[int] = set()
    unique = []
    duplicates = 0
    for ex in examples:
        key_tuple = tuple(str(ex.get(field, "")).strip().lower() for field in key_fields)
        key_hash = hash(key_tuple)
        if key_hash not in seen:
            seen.add(key_hash)
            unique.append(ex)
        else:
            duplicates += 1
    logger.info(f"✓ Deduplication: {len(examples)} → {len(unique)} examples ({duplicates} removed)")
    return unique, duplicates

def map_correction_format(corrections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert corrections.jsonl format"""
    mapped = []
    for corr in corrections:
        mapped.append({
            "question": corr.get("question", ""),
            "sql_gold": corr.get("corrected_sql", ""),
            "source": "corrections",
            "quality_tier": "high"
        })
    return mapped

def map_training_format(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert training_data_sets format"""
    mapped = []
    for ex in examples:
        question = ex.get("input") or ex.get("question") or ""
        sql = ex.get("output") or ex.get("sql") or ""
        if question and sql:
            mapped.append({
                "question": question.strip(),
                "sql_gold": normalize_sql(sql),
                "source": "main_training",
                "quality_tier": "primary"
            })
    return mapped

def find_join_paths(tables: List[str], joins: List[str]) -> List[str]:
    """Find join paths between tables"""
    if len(tables) <= 1:
        return []
    relevant_joins = []
    tables_upper = [t.upper() for t in tables]
    for join in joins:
        if "=" in join:
            left, right = join.split("=")
            left_table = left.split(".")[0].strip().upper()
            right_table = right.split(".")[0].strip().upper()
            if left_table in tables_upper and right_table in tables_upper:
                relevant_joins.append(join.strip())
    return relevant_joins

def build_sqlcoder_schema_context(
    tables: List[str],
    joins: List[str],
    schemas: Dict[str, Dict[str, Any]]
) -> str:
    """Generate SQLCoder-style schema context (DDL-like)"""
    schema_parts = []
    for table in tables:
        if table in schemas:
            schema = schemas[table]
            cols_info = []
            columns_data = schema.get("columns", [])
            if isinstance(columns_data, list):
                for col in columns_data:
                    cols_info.append(f"    {col.get('name')} {col.get('type', 'VARCHAR2(255)')}")
            else:
                for name, info in columns_data.items():
                    dtype = info.get('type', 'VARCHAR2(255)') if isinstance(info, dict) else 'VARCHAR2(255)'
                    cols_info.append(f"    {name} {dtype}")
            
            table_ddl = f"CREATE TABLE {table} (\n" + ",\n".join(cols_info) + "\n);"
            schema_parts.append(table_ddl)
        else:
            schema_parts.append(f"-- Table {table} schema not available")
    
    # Add joins as comments
    join_paths = find_join_paths(tables, joins)
    if join_paths:
        schema_parts.append("-- Join relationships:")
        for jp in join_paths:
            schema_parts.append(f"-- {jp}")
            
    return "\n\n".join(schema_parts)

# ============================================================================
# PHASE 2.5: AUGMENTATION
# ============================================================================

def augment_schema_permutation(
    example: Dict[str, Any],
    schema_context: str,
    num_variations: int = 1
) -> List[Dict[str, Any]]:
    """Generate schema permutation variations for SQLCoder"""
    # For SQLCoder, we can shuffle the order of CREATE TABLE statements
    parts = schema_context.split("\n\n")
    table_parts = [p for p in parts if p.startswith("CREATE TABLE")]
    other_parts = [p for p in parts if not p.startswith("CREATE TABLE")]
    
    if len(table_parts) <= 1:
        return []
        
    variations = []
    for _ in range(num_variations):
        shuffled_tables = table_parts.copy()
        random.shuffle(shuffled_tables)
        new_schema = "\n\n".join(shuffled_tables + other_parts)
        if new_schema != schema_context:
            variation = example.copy()
            variation["_augmentation_type"] = "schema_permutation"
            variation["_schema_context"] = new_schema
            variations.append(variation)
    return variations

# ============================================================================
# PHASE 3: FORMAT CONVERSION
# ============================================================================

def create_sqlcoder_conversation(
    question: str,
    sql_gold: str,
    schema_context: str
) -> List[Dict[str, str]]:
    """Create SQLCoder-formatted conversation"""
    user_prompt = make_sqlcoder_prompt(question, schema_context)
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": sql_gold.strip()}
    ]

def build_training_example(
    example_id: str,
    question: str,
    sql_gold: str,
    schema_context: str,
    tables: List[str],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Build complete training example"""
    conversations = create_sqlcoder_conversation(question, sql_gold, schema_context)
    return {
        "id": example_id,
        "sql_type": "oracle",
        "conversations": conversations,
        "meta": {
            **metadata,
            "tables": tables,
            "schema_length": len(schema_context),
            "prompt_token_estimate": len(conversations[0]["content"]) // 4,
            "sql_length": len(sql_gold)
        }
    }

# ============================================================================
# PHASE 4: VALIDATION
# ============================================================================

def validate_sql(sql: str) -> Tuple[bool, List[str]]:
    """Validate SQL content"""
    errors = []
    if not sql:
        errors.append("SQL is empty")
        return False, errors
    if "```" in sql:
        errors.append("SQL contains markdown fences")
        return False, errors
    if not re.match(r"^\s*(SELECT|WITH)", sql, re.IGNORECASE):
        errors.append("SQL must start with SELECT or WITH")
        return False, errors
    if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b", sql, re.IGNORECASE):
        errors.append("SQL contains DML/DDL statements")
        return False, errors
    return True, errors

def validate_example(
    example: Dict[str, Any],
    schemas: Dict[str, Dict[str, Any]]
) -> Tuple[bool, List[str], List[str]]:
    """Validate training example"""
    errors = []
    warnings = []
    if "conversations" not in example:
        errors.append("Missing conversations field")
        return False, errors, warnings
    
    sql_content = example["conversations"][1]["content"]
    sql_valid, sql_errors = validate_sql(sql_content)
    errors.extend(sql_errors)
    
    tables = extract_tables_from_sql(sql_content)
    for table in tables:
        if table not in schemas:
            warnings.append(f"Table {table} not in schemas")
            
    return len(errors) == 0, errors, warnings

# ============================================================================
# ORCHESTRATION
# ============================================================================

def build_dataset_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Orchestrate SQLCoder dataset build"""
    logger.info("=" * 80)
    logger.info("ORACLE EBS NL2SQL DATASET PIPELINE (SQLCODER)")
    logger.info("=" * 80)
    
    main_data = load_main_training_data(config["input"]["main_training"])
    corrections = load_corrections(config["input"]["corrections"])
    joins = load_joins(config["input"]["joins"])
    schemas = load_table_schemas(config["input"]["table_schemas"])
    keywords = load_keywords(config["input"]["keywords"])
    
    main_mapped = map_training_format(main_data)
    corr_mapped = map_correction_format(corrections)
    
    combined = main_mapped + corr_mapped
    combined, _ = deduplicate_examples(combined)
    
    examples_with_context = []
    for ex in combined:
        tables = extract_tables_from_sql(ex.get("sql_gold", ""))
        schema_context = build_sqlcoder_schema_context(tables, joins, schemas)
        ex["_schema_context"] = schema_context
        ex["_tables"] = tables
        examples_with_context.append(ex)
        
    augmented_examples = []
    aug_config = config.get("augmentation", {})
    if aug_config.get("enabled"):
        for ex in examples_with_context:
            augmented_examples.append(ex)
            if aug_config.get("techniques", {}).get("schema_permutation", {}).get("enabled"):
                variations = augment_schema_permutation(
                    ex, 
                    ex["_schema_context"],
                    num_variations=aug_config.get("techniques", {}).get("schema_permutation", {}).get("variations_per_example", 1)
                )
                augmented_examples.extend(variations)
    else:
        augmented_examples = examples_with_context
        
    sqlcoder_examples = []
    for i, ex in enumerate(augmented_examples):
        try:
            askguru_ex = build_training_example(
                f"oracle_sqlcoder_v1_{i:06d}",
                ex["question"],
                ex["sql_gold"],
                ex["_schema_context"],
                ex["_tables"],
                {
                    "source": ex.get("source", "unknown"),
                    "quality_tier": ex.get("quality_tier", "primary"),
                    "augmentation_type": ex.get("_augmentation_type", "original")
                }
            )
            sqlcoder_examples.append(askguru_ex)
        except Exception as e:
            logger.warning(f"Failed to convert example {i}: {e}")

    valid_examples = []
    for ex in sqlcoder_examples:
        valid, _, _ = validate_example(ex, schemas)
        if valid:
            valid_examples.append(ex)
            
    random.shuffle(valid_examples)
    train_ratio = config.get("splits", {}).get("train_ratio", 0.8)
    val_ratio = config.get("splits", {}).get("val_ratio", 0.1)
    
    t_idx = int(len(valid_examples) * train_ratio)
    v_idx = t_idx + int(len(valid_examples) * val_ratio)
    
    return {
        "train": valid_examples[:t_idx],
        "val": valid_examples[t_idx:v_idx],
        "test": valid_examples[v_idx:],
        "full": valid_examples
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    result = build_dataset_from_config(config)
    
    output_dir = config["output"]["base_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    for split, data in [("train", result["train"]), ("val", result["val"]), ("test", result["test"]), ("full", result["full"])]:
        path = os.path.join(output_dir, f"oracle_sqlcoder_sft_{split}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"✓ Wrote {len(data)} to {path}")

if __name__ == "__main__":
    main()
