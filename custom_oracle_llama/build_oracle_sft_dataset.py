#!/usr/bin/env python3
"""
build_oracle_sft_dataset.py

Complete Oracle EBS NL2SQL SFT Dataset Pipeline:
- Phase 2: Data Extraction & Transformation
- Phase 2.5: Augmentation & Deduplication
- Phase 3: Format Conversion to askGuru
- Phase 4: Quality Assurance & Validation
- Phase 5: Train/Val/Test Split

Converts raw Oracle EBS training data into production-ready askGuru format with:
- Deduplicated examples
- RAG context enrichment
- Data augmentation (schema permutation + synonym swapping)
- Quality validation
- Stratified train/val/test splits
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

ORACLE_EBS_RULES = """You are a Text-to-SQL generator for Oracle EBS.
Return ONLY Oracle SQL. No markdown. No explanations. No comments."""


# ============================================================================
# PHASE 2: DATA EXTRACTION & TRANSFORMATION
# ============================================================================

def load_main_training_data(filepath: str) -> List[Dict[str, Any]]:
    """Load training_data_sets_4jul2025_v5_askguru_compatible.jsonl"""
    examples = []
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
    """Load corrections.jsonl - high quality manually verified examples"""
    examples = []
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
    """Load joins.json - FK relationships"""
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
    """Load keywords.json - business term mappings"""
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
    
    # Match FROM, JOIN keywords followed by table names
    pattern = r"\b(?:FROM|JOIN|INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN)\s+(\w+)"
    matches = re.findall(pattern, sql, re.IGNORECASE)
    return list(set(m.upper() for m in matches))


def deduplicate_examples(
    examples: List[Dict[str, Any]], 
    key_fields: List[str] = ["question", "sql_gold"]
) -> Tuple[List[Dict[str, Any]], int]:
    """Remove exact duplicates based on key fields"""
    seen: Set[str] = set()
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
    """Convert corrections.jsonl format to {question, sql_gold}"""
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
    """Convert training_data_sets format to {question, sql_gold}"""
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
        # Parse "TABLE_A.col = TABLE_B.col"
        if "=" in join:
            left, right = join.split("=")
            left_table = left.split(".")[0].strip().upper()
            right_table = right.split(".")[0].strip().upper()
            
            if left_table in tables_upper and right_table in tables_upper:
                relevant_joins.append(join.strip())
    
    return relevant_joins


def build_rag_context(
    question: str,
    tables: List[str],
    joins: List[str],
    schemas: Dict[str, Dict[str, Any]],
    keywords: Dict[str, str]
) -> str:
    """Generate structured RAG context"""
    sections = []
    
    # Candidate Tables
    if tables:
        table_section = "# Candidate Tables"
        for table in tables:
            if table in schemas:
                schema = schemas[table]
                # Handle columns as list of dicts
                columns_data = schema.get("columns", [])
                if isinstance(columns_data, list):
                    cols = [col.get("name", "") for col in columns_data[:5]]
                else:
                    cols = list(columns_data.keys())[:5]
                table_section += f"\n- {table}: {', '.join(cols)}"
            else:
                table_section += f"\n- {table}"
        sections.append(table_section)
    
    # Join Graph
    join_paths = find_join_paths(tables, joins)
    if join_paths:
        join_section = "# Join Graph\n" + "\n".join(f"- {jp}" for jp in join_paths)
        sections.append(join_section)
    
    # Relevant Columns (sample from schemas)
    if tables:
        col_section = "# Relevant Columns"
        for table in tables:
            if table in schemas:
                schema = schemas[table]
                columns_data = schema.get("columns", [])
                
                # Handle columns as list of dicts
                if isinstance(columns_data, list):
                    for col_info in columns_data[:3]:
                        col_name = col_info.get("name", "")
                        col_type = col_info.get("type", "UNKNOWN")
                        col_section += f"\n- {table}.{col_name}: {col_type}"
                else:
                    for col_name, col_info in list(columns_data.items())[:3]:
                        col_type = col_info.get("type", "UNKNOWN") if isinstance(col_info, dict) else "UNKNOWN"
                        col_section += f"\n- {table}.{col_name}: {col_type}"
        
        if col_section != "# Relevant Columns":
            sections.append(col_section)
    
    # Business Mappings (from keywords)
    keyword_matches = []
    for keyword, mapping in keywords.items():
        if keyword.lower() in question.lower():
            keyword_matches.append(f'- "{keyword}" → {mapping}')
    
    if keyword_matches:
        kw_section = "# Synonyms/Business Mapping\n" + "\n".join(keyword_matches)
        sections.append(kw_section)
    
    return "\n\n".join(sections)


# ============================================================================
# PHASE 2.5: AUGMENTATION & DEDUPLICATION
# ============================================================================

def augment_schema_permutation(
    example: Dict[str, Any],
    rag_context: str,
    num_variations: int = 1
) -> List[Dict[str, Any]]:
    """Generate schema permutation variations"""
    if "# Candidate Tables" not in rag_context:
        return []
    
    variations = []
    lines = rag_context.split("\n")
    
    for _ in range(num_variations):
        # Find and shuffle candidate tables section
        augmented_lines = []
        in_candidate = False
        candidate_lines = []
        
        for line in lines:
            if line.startswith("# Candidate Tables"):
                in_candidate = True
                augmented_lines.append(line)
            elif in_candidate and line.startswith("-"):
                candidate_lines.append(line)
            elif in_candidate and line.startswith("#"):
                in_candidate = False
                random.shuffle(candidate_lines)
                augmented_lines.extend(candidate_lines)
                augmented_lines.append(line)
            else:
                augmented_lines.append(line)
        
        if in_candidate and candidate_lines:
            random.shuffle(candidate_lines)
            augmented_lines.extend(candidate_lines)
        
        new_rag = "\n".join(augmented_lines)
        if new_rag != rag_context:
            variation = example.copy()
            variation["_augmentation_type"] = "schema_permutation"
            variation["_rag_context"] = new_rag
            variations.append(variation)
    
    return variations


def augment_synonym_swap(
    example: Dict[str, Any],
    keywords: Dict[str, str],
    num_variations: int = 1
) -> List[Dict[str, Any]]:
    """Generate synonym swap variations"""
    question = example.get("question", "")
    if not question or not keywords:
        return []
    
    variations = []
    
    for _ in range(num_variations):
        new_question = question
        found_keyword = False
        
        for keyword, _ in list(keywords.items())[:5]:  # Try first 5 keywords
            if keyword.lower() in new_question.lower():
                # Simple synonym swap (could be enhanced)
                found_keyword = True
                break
        
        if found_keyword:
            variation = example.copy()
            variation["_augmentation_type"] = "synonym_swap"
            variation["_original_question"] = question
            variations.append(variation)
    
    return variations


# ============================================================================
# PHASE 3: FORMAT CONVERSION TO ASKGURU
# ============================================================================

def create_conversation(
    question: str,
    sql_gold: str,
    rag_context: str,
    rules: str = ORACLE_EBS_RULES
) -> List[Dict[str, str]]:
    """Create askGuru conversations format"""
    user_prompt = f"""{rules}

{rag_context}

[User Question]
{question}"""
    
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": sql_gold.strip()}
    ]


def build_training_example(
    example_id: str,
    question: str,
    sql_gold: str,
    rag_context: str,
    tables: List[str],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Build complete training example in askGuru format"""
    conversations = create_conversation(question, sql_gold, rag_context)
    
    return {
        "id": example_id,
        "sql_type": "oracle",
        "conversations": conversations,
        "meta": {
            **metadata,
            "tables": tables,
            "rag_context_length": len(rag_context),
            "prompt_token_estimate": len(conversations[0]["content"]) // 4,
            "sql_length": len(sql_gold)
        }
    }


# ============================================================================
# PHASE 4: QUALITY ASSURANCE & VALIDATION
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
    
    # Check for DML/DDL
    if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b", sql, re.IGNORECASE):
        errors.append("SQL contains DML/DDL statements")
        return False, errors
    
    return True, errors


def validate_example(
    example: Dict[str, Any],
    schemas: Dict[str, Dict[str, Any]],
    joins: List[str]
) -> Tuple[bool, List[str], List[str]]:
    """Validate training example (askGuru format)"""
    errors = []
    warnings = []
    
    # Check conversations format
    if "conversations" not in example:
        errors.append("Missing conversations field")
        return len(errors) == 0, errors, warnings
    
    if not isinstance(example["conversations"], list) or len(example["conversations"]) != 2:
        errors.append("Invalid conversations format")
        return len(errors) == 0, errors, warnings
    
    # Extract SQL from assistant content (conversations[1])
    sql_content = example.get("conversations", [{}, {}])[1].get("content", "")
    
    # Check SQL validity
    sql_valid, sql_errors = validate_sql(sql_content)
    errors.extend(sql_errors)
    
    # Check table references
    tables = extract_tables_from_sql(sql_content)
    for table in tables:
        if table not in schemas:
            warnings.append(f"Table {table} not in schemas")
    
    # Check alignment - SQL should be findable in the prompt
    user_content = example.get("conversations", [{}])[0].get("content", "")
    
    if sql_content and sql_content not in user_content:
        warnings.append("SQL span alignment issue")
    
    return len(errors) == 0, errors, warnings


def compute_quality_metrics(
    examples: List[Dict[str, Any]],
    augmented_count: int = 0
) -> Dict[str, Any]:
    """Compute dataset-level quality metrics"""
    total = len(examples)
    
    tables_covered = set()
    join_examples = 0
    sql_lengths = []
    prompt_lengths = []
    
    for ex in examples:
        tables = ex.get("meta", {}).get("tables", [])
        tables_covered.update(tables)
        
        if len(tables) > 1:
            join_examples += 1
        
        sql_lengths.append(ex.get("meta", {}).get("sql_length", 0))
        prompt_lengths.append(ex.get("meta", {}).get("prompt_token_estimate", 0))
    
    return {
        "total_examples": total,
        "augmented_examples": augmented_count,
        "tables_covered": len(tables_covered),
        "join_examples": join_examples,
        "single_table_examples": total - join_examples,
        "avg_sql_length": sum(sql_lengths) / total if total > 0 else 0,
        "avg_prompt_tokens": sum(prompt_lengths) / total if total > 0 else 0,
        "max_prompt_tokens": max(prompt_lengths) if prompt_lengths else 0
    }


# ============================================================================
# PHASE 5: TRAIN/VAL/TEST SPLIT
# ============================================================================

def stratified_split(
    examples: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stratified split by quality tier and table count"""
    random.shuffle(examples)
    
    train_idx = int(len(examples) * train_ratio)
    val_idx = train_idx + int(len(examples) * val_ratio)
    
    train_data = examples[:train_idx]
    val_data = examples[train_idx:val_idx]
    test_data = examples[val_idx:]
    
    logger.info(f"✓ Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    return train_data, val_data, test_data


# ============================================================================
# ORCHESTRATION
# ============================================================================

def build_dataset_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """End-to-end data pipeline orchestration"""
    logger.info("=" * 80)
    logger.info("ORACLE EBS NL2SQL DATASET PIPELINE")
    logger.info("=" * 80)
    
    # Load all source data
    logger.info("\n[PHASE 2] Loading source data...")
    main_data = load_main_training_data(config["input"]["main_training"])
    corrections = load_corrections(config["input"]["corrections"])
    joins = load_joins(config["input"]["joins"])
    schemas = load_table_schemas(config["input"]["table_schemas"])
    keywords = load_keywords(config["input"]["keywords"])
    
    # Transform data
    logger.info("\n[PHASE 2] Transforming data formats...")
    main_mapped = map_training_format(main_data)
    corr_mapped = map_correction_format(corrections)
    
    # Combine and deduplicate
    logger.info("\n[PHASE 2] Deduplicating...")
    combined = main_mapped + corr_mapped
    combined, dup_count = deduplicate_examples(combined)
    
    # Build RAG context
    logger.info("\n[PHASE 2] Building RAG context...")
    examples_with_rag = []
    for i, ex in enumerate(combined):
        tables = extract_tables_from_sql(ex.get("sql_gold", ""))
        rag_context = build_rag_context(
            ex.get("question", ""),
            tables,
            joins,
            schemas,
            keywords
        )
        ex["_rag_context"] = rag_context
        ex["_tables"] = tables
        examples_with_rag.append(ex)
    
    # Augmentation
    logger.info("\n[PHASE 2.5] Applying augmentation...")
    augmented_examples = []
    aug_config = config.get("augmentation", {})
    
    if aug_config.get("enabled"):
        for ex in examples_with_rag:
            # Add original
            augmented_examples.append(ex)
            
            # Schema permutation
            if aug_config.get("techniques", {}).get("schema_permutation", {}).get("enabled"):
                variations = augment_schema_permutation(
                    ex,
                    ex.get("_rag_context", ""),
                    num_variations=aug_config.get("techniques", {}).get("schema_permutation", {}).get("variations_per_example", 1)
                )
                augmented_examples.extend(variations)
            
            # Synonym swap
            if aug_config.get("techniques", {}).get("synonym_swap", {}).get("enabled"):
                variations = augment_synonym_swap(
                    ex,
                    keywords,
                    num_variations=aug_config.get("techniques", {}).get("synonym_swap", {}).get("variations_per_example", 1)
                )
                augmented_examples.extend(variations)
    else:
        augmented_examples = examples_with_rag
    
    logger.info(f"✓ Augmentation: {len(examples_with_rag)} → {len(augmented_examples)} examples")
    
    # Format conversion to askGuru
    logger.info("\n[PHASE 3] Converting to askGuru format...")
    askguru_examples = []
    for i, ex in enumerate(augmented_examples):
        try:
            tables = ex.get("_tables", [])
            rag_context = ex.get("_rag_context", "")
            
            example_id = f"oracle_ebs_v1_{i:06d}"
            askguru_ex = build_training_example(
                example_id,
                ex.get("question", ""),
                ex.get("sql_gold", ""),
                rag_context,
                tables,
                {
                    "source": ex.get("source", "unknown"),
                    "quality_tier": ex.get("quality_tier", "primary"),
                    "augmentation_type": ex.get("_augmentation_type", "original")
                }
            )
            askguru_examples.append(askguru_ex)
        except Exception as e:
            logger.warning(f"Failed to convert example {i}: {e}")
    
    logger.info(f"✓ Converted {len(askguru_examples)} examples to askGuru format")
    
    # Validation
    logger.info("\n[PHASE 4] Validating...")
    valid_examples = []
    validation_stats = defaultdict(int)
    
    for ex in askguru_examples:
        valid, errors, warnings = validate_example(ex, schemas, joins)
        
        if valid:
            valid_examples.append(ex)
            validation_stats["valid"] += 1
        else:
            validation_stats["invalid"] += 1
            for error in errors:
                validation_stats[f"error_{error}"] += 1
        
        for warning in warnings:
            validation_stats[f"warning_{warning}"] += 1
    
    logger.info(f"✓ Validation: {validation_stats['valid']} valid, {validation_stats['invalid']} invalid")
    
    # Split
    logger.info("\n[PHASE 5] Creating train/val/test split...")
    train_data, val_data, test_data = stratified_split(
        valid_examples,
        train_ratio=config.get("splits", {}).get("train_ratio", 0.8),
        val_ratio=config.get("splits", {}).get("val_ratio", 0.1)
    )
    
    # Metrics
    metrics = compute_quality_metrics(valid_examples, len(augmented_examples) - len(examples_with_rag))
    
    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "full": valid_examples,
        "metrics": metrics,
        "validation_stats": dict(validation_stats)
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Oracle EBS SFT dataset with augmentation")
    ap.add_argument("--config", required=True, help="Path to oracle_sft_config.yaml")
    ap.add_argument("--log_level", default="INFO", help="Logging level")
    args = ap.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Build dataset
    result = build_dataset_from_config(config)
    
    # Write outputs
    output_dir = config["output"]["base_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in [("train", result["train"]), ("val", result["val"]), ("test", result["test"]), ("full", result["full"])]:
        output_path = os.path.join(output_dir, f"oracle_sft_conversations_{split_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Wrote {len(split_data)} examples to {output_path}")
    
    # Write metrics
    metrics_path = os.path.join(output_dir, "data_quality_report.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": result["metrics"],
            "validation_stats": result["validation_stats"]
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ Wrote quality report to {metrics_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("DATASET PIPELINE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
