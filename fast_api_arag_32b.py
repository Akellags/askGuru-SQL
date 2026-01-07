
# -*- coding: utf-8 -*-
"""
fast_api_arag_fixed.py — FastAPI for Agentic RAG → XiYan → SQL
- Keeps behavior aligned with run_local_7b.py
- Adds post-generation SQL guard (patch-only by default)
- Adds a corrections feedback loop (store -> retrieve as prioritized few-shots)
- No business hardcoding; paths/config via JSON or env

Run:
  uvicorn fast_api_arag_32b:app --host 0.0.0.0 --port 8000

Optional config:
  export ASK_GURU_CONFIG=/askGuruARAG/rag/config/server_config.json
  (Same shape as the refined server; falls back to defaults if not set)
"""
import os, sys, json, re, tempfile, subprocess, time, logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- CUDA Optimization Settings ----------
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["PYTORCH_DISABLE_DISTRIBUTED_TENSOR"] = "1"
os.environ["TORCH_DIST_TENSOR_DISABLE"] = "1"

# ---------- Environment (HF caches) ----------
HF_HOME = os.environ.get("HF_HOME", "/LayoutLMtraining/askGuruARAG/.hf")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{HF_HOME}/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{HF_HOME}/hub")
os.environ.setdefault("HF_DATASETS_CACHE", f"{HF_HOME}/datasets")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

HERE = Path(__file__).resolve()
BASE = HERE.parent  # assume placed in rag/src

# ---------- Model Path Configuration ----------
# HuggingFace model ID for XiYanSQL-QwenCoder-32B-2504
MODEL_PATH = os.environ.get("MODEL_PATH", "XGenerationLab/XiYanSQL-QwenCoder-32B-2504")

# ---------- Local modules ----------
from planner import plan, load_cards
from router import route_tables
from ood_gate import gate_or_message
from llm_utils import GenConfig, extract_sql

# Optional authoring helper
try:
    from prompt_author_7b import author_prompt_with_7b
except Exception:
    author_prompt_with_7b = None

# Optional embedder for semantic similarity marking
try:
    from dense_utils import Embedder
    import numpy as np
    _EMBEDDER_AVAILABLE = True
except Exception as e:
    logger.warning(f"Embedder not available: {e}. Similarity marking will be disabled.")
    _EMBEDDER_AVAILABLE = False

SQL_GUARD = BASE / "sql_guard.py"

# ---------- Config ----------
@dataclass
class APIConfig:
    # model - Using XiYanSQL-QwenCoder-32B from HuggingFace
    model_id: str = MODEL_PATH  # HF model ID: XGenerationLab/XiYanSQL-QwenCoder-32B-2504
    max_new_tokens: int = 1500
    temperature: float = 0.0
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    prompt_mode: str = "xiyan"  # xiyan | author7b

    # paths
    index_dir: str = str((BASE.parent / "index" / "bm25").resolve())
    vectors_dir: str = str((BASE.parent / "index" / "vectors").resolve())
    tables_dir: str = str((BASE.parent / "data" / "tables").resolve())
    fewshots: str = str((BASE.parent / "data" / "fewshots.jsonl").resolve())
    joins: str = str((BASE.parent / "data" / "joins" / "joins.json").resolve())
    router_map: str = str((BASE.parent / "data" / "router" / "keywords.json").resolve())
    feedback_store: str = str((BASE.parent / "data" / "feedback" / "corrections.jsonl").resolve())

    # retrieval
    max_tables: int = 8
    per_table_cols: int = 1000
    fewshots_k: int = 6

    # similarity (for marking examples)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_exact_threshold: float = 0.80
    similarity_close_threshold: float = 0.60

    # guard
    guard_policy: str = "patch-only"  # detect-only | patch-only | retry-then-patch
    
    # memory optimization - set to False to unload model after each inference (prevents OOM)
    cache_model: bool = True  # If False, model is completely unloaded after each inference

def load_api_config() -> APIConfig:
    path = os.environ.get("ASK_GURU_CONFIG", "")
    cfg = {}
    if path and Path(path).exists():
        try:
            cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    c = APIConfig()
    g = cfg.get("generation", {})
    p = cfg.get("paths", {})
    r = cfg.get("retrieval", {})
    gd = cfg.get("guard", {})

    c.model_id = g.get("model_id", c.model_id)
    c.max_new_tokens = int(g.get("max_new_tokens", c.max_new_tokens))
    c.temperature = float(g.get("temperature", c.temperature))
    c.top_p = float(g.get("top_p", c.top_p))
    c.repetition_penalty = float(g.get("repetition_penalty", c.repetition_penalty))
    c.prompt_mode = g.get("prompt_mode", c.prompt_mode)

    c.index_dir = p.get("index_dir", c.index_dir)
    c.vectors_dir = p.get("vectors_dir", c.vectors_dir)
    c.tables_dir = p.get("tables_dir", c.tables_dir)
    c.fewshots = p.get("fewshots", c.fewshots)
    c.joins = p.get("joins", c.joins)
    c.router_map = p.get("router_map", c.router_map)
    c.feedback_store = p.get("feedback_store", c.feedback_store)

    c.max_tables = int(r.get("max_tables", c.max_tables))
    c.per_table_cols = int(r.get("per_table_cols", c.per_table_cols))
    c.fewshots_k = int(r.get("fewshots_k", c.fewshots_k))

    sim = cfg.get("similarity", {})
    c.embedding_model = sim.get("embedding_model", c.embedding_model)
    c.similarity_exact_threshold = float(sim.get("exact_threshold", c.similarity_exact_threshold))
    c.similarity_close_threshold = float(sim.get("close_threshold", c.similarity_close_threshold))

    c.guard_policy = gd.get("policy", c.guard_policy)
    
    # memory optimization config
    mem = cfg.get("memory", {})
    c.cache_model = bool(mem.get("cache_model", c.cache_model))
    
    return c

CFG = load_api_config()

# ---------- Global Embedder (for similarity marking) ----------
_GLOBAL_EMBEDDER: Optional[Any] = None

def get_embedder():
    """Lazy-load the embedder for semantic similarity marking.
    
    Loads only on first request to preserve GPU memory.
    """
    global _GLOBAL_EMBEDDER
    if not _EMBEDDER_AVAILABLE:
        return None
    if _GLOBAL_EMBEDDER is None:
        try:
            logger.info(f"🔧 Loading embedder: {CFG.embedding_model}")
            _GLOBAL_EMBEDDER = Embedder(CFG.embedding_model)
            logger.info(f"✅ Embedder loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load embedder: {e}")
            return None
    return _GLOBAL_EMBEDDER

def cleanup_embedder():
    """Unload embedder from GPU to free memory immediately after use."""
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is not None:
        try:
            _GLOBAL_EMBEDDER.cleanup()
            logger.info("🧹 Embedder cleaned up from GPU")
        except Exception as e:
            logger.warning(f"Failed to cleanup embedder: {e}")

def batch_compute_similarities(user_question: str, intents: List[str]) -> List[float]:
    """Compute similarities for multiple intents in a batch for efficiency.
    
    This batches the encoding operation to avoid repeated GPU loading.
    Returns similarities in the same order as input intents.
    """
    if not user_question or not intents:
        return [0.0] * len(intents)
    
    embedder = get_embedder()
    if embedder is None:
        return [0.0] * len(intents)
    
    try:
        # Batch encode all intents at once
        all_texts = [user_question] + intents
        embeddings = embedder.encode_batch(all_texts, prefix="")
        
        user_emb = embeddings[0]
        similarities = []
        
        for intent_emb in embeddings[1:]:
            # Cosine similarity (embeddings already normalized)
            dot = np.dot(user_emb, intent_emb)
            similarities.append(float(dot))
        
        return similarities
    except Exception as e:
        logger.warning(f"Batch similarity computation failed: {e}")
        return [0.0] * len(intents)

def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts using the global embedder.
    
    Note: For multiple comparisons, use batch_compute_similarities instead for efficiency.
    """
    embedder = get_embedder()
    if embedder is None:
        return 0.0
    try:
        emb1 = embedder.encode_query(text1, prefix="")[0]
        emb2 = embedder.encode_query(text2, prefix="")[0]
        dot = np.dot(emb1, emb2)
        return float(dot)
    except Exception as e:
        logger.warning(f"Similarity computation failed: {e}")
        return 0.0

# ---------- Data helpers ----------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        try:
            items.append(json.loads(line))
        except Exception:
            continue
    return items

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def build_db_schema(plan_dict: Dict[str, Any], tables_dir: Path) -> str:
    cards = load_cards(tables_dir)
    out_lines: List[str] = []
    for t in plan_dict.get("tables_selected", []):
        T = t.upper()
        card = cards.get(T, {})
        type_map = { str(c.get("name","")).upper(): (str(c.get("type","")).upper(), str(c.get("desc","") or c.get("description","")).strip())
                     for c in card.get("columns", []) }
        essential_col_names = {str(ec.get("name","")).upper() for ec in (card.get("essential_columns") or [])}
        essential_reasons = {str(ec.get("name","")).upper(): str(ec.get("reason","")).strip() for ec in (card.get("essential_columns") or [])}
        cols = plan_dict.get("columns", {}).get(T, [])
        out_lines.append(f"{T} (")
        for name in cols:
            col_upper = name.upper()
            is_essential = col_upper in essential_col_names
            marker = "**[ESSENTIAL]** " if is_essential else ""
            meta = type_map.get(col_upper)
            reason = essential_reasons.get(col_upper, "")
            if meta:
                if is_essential and reason:
                    out_lines.append(f"  {marker}{name} {meta[0]} -- {reason}")
                else:
                    out_lines.append(f"  {marker}{name} {meta[0]} -- {meta[1]}")
            else:
                out_lines.append(f"  {marker}{name}")
        out_lines.append(")\n")
    jh = plan_dict.get("join_conds", [])
    if jh:
        out_lines.append("JOIN HINTS:")
        for j in jh:
            out_lines.append(f"  - {j}")
    return "\n".join(out_lines).strip()

def _load_fewshot_map(fewshots_path: Path) -> Dict[str, Dict[str, Any]]:
    fmap: Dict[str, Dict[str, Any]] = {}
    path = Path(fewshots_path)
    if not path.exists():
        return fmap
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
        except Exception:
            continue
        if "id" in ex:
            fmap[str(ex["id"])] = ex
    return fmap

def inject_examples_block(selected_examples: List[Dict[str, Any]], fewshots_path: Path, mode: str = "sql", max_examples: int = 2, max_sql_chars: int = 4000, user_question: str = "") -> str:
    if not selected_examples:
        return ""
    
    fmap = _load_fewshot_map(fewshots_path)
    
    # Prepare intents for batch similarity computation
    intents_with_idx = []
    for k, ex in enumerate(selected_examples):
        if k >= max_examples:
            break
        ex_id = str(ex.get("id", k+1))
        full = fmap.get(ex_id, {})
        intent = (full.get("intent", ex.get("intent","")) or "").replace("\n"," ").strip()
        if user_question and intent:
            intents_with_idx.append((k, intent))
    
    # Batch compute similarities for all intents at once (more efficient GPU usage)
    similarities_dict = {}
    if intents_with_idx:
        intents = [intent for _, intent in intents_with_idx]
        similarities = batch_compute_similarities(user_question, intents)
        for (k, intent), sim in zip(intents_with_idx, similarities):
            similarities_dict[k] = sim
    
    # Build output with precomputed similarities
    lines = ["### Few Shot examples"]
    k = 0
    for ex in selected_examples:
        if k >= max_examples:
            break
        ex_id = str(ex.get("id", k+1))
        full = fmap.get(ex_id, {})
        intent = (full.get("intent", ex.get("intent","")) or "").replace("\n"," ").strip()
        tabs = [t.upper() for t in full.get("tables", ex.get("tables", []))]
        sql = (full.get("sql","") or ex.get("sql","") or "").strip()
        
        # Add marker based on precomputed similarity
        marker = ""
        if k in similarities_dict:
            similarity = similarities_dict[k]
            if similarity >= CFG.similarity_exact_threshold:
                marker = " [* EXACT MATCH]"
                logger.info(f"   🎯 Example {k+1} marked as EXACT MATCH (similarity: {similarity:.2%})")
            elif similarity >= CFG.similarity_close_threshold:
                marker = " [* VERY SIMILAR]"
                logger.info(f"   ⭐ Example {k+1} marked as VERY SIMILAR (similarity: {similarity:.2%})")
        
        if intent:
            lines.append(f"Question {k+1}{marker}: {intent}")
        if tabs:
            lines.append(f"Tables: {', '.join(tabs)}")
        if mode == "sql" and sql:
            if len(sql) > max_sql_chars:
                sql = sql[:max_sql_chars] + "\n-- [truncated]"
            lines.append("SQL:")
            lines.append(sql)
        k += 1
    
    lines.append("### End of Few Shot examples")
    result = "\n".join(lines).strip()
    
    # Cleanup embedder after use to free GPU memory
    cleanup_embedder()
    
    return result

# ---------- Corrections selection ----------
def parse_tables_from_sql(sql: str) -> List[str]:
    # very simple extractor; relies on FROM and JOIN tokens
    toks = re.findall(r'\bfrom\s+([a-zA-Z0-9_#\$]+)|\bjoin\s+([a-zA-Z0-9_#\$]+)', sql, flags=re.IGNORECASE)
    tabs = set()
    for a,b in toks:
        t = (a or b)
        if t: tabs.add(t.upper())
    return list(tabs)

def select_priority_corrections(question: str, plan_dict: Dict[str, Any], store_path: Path, k: int = 2) -> List[Dict[str, Any]]:
    corr = load_jsonl(store_path)
    if not corr: return []
    want = set([t.upper() for t in plan_dict.get("tables_selected", [])])
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for ex in corr:
        sql = ex.get("corrected_sql","") or ex.get("sql","")
        tables = ex.get("tables") or parse_tables_from_sql(sql)
        inter = len(want.intersection([t.upper() for t in tables]))
        bonus = 1.0 if ex.get("question","").strip().lower() in (question.strip().lower(),) else 0.0
        score = inter + bonus
        if score > 0:
            scored.append((score, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:k]]

# ---------- Guard ----------
def run_sql_guard(sql: str, tables_dir: Path, patch: bool = True) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile("w+", suffix=".sql", delete=False) as f_in, \
         tempfile.NamedTemporaryFile("r", suffix=".sql", delete=False) as f_out:
        f_in.write(sql); f_in.flush()
        cmd = [sys.executable, str(SQL_GUARD),
               "--sql-file", f_in.name,
               "--oracle-syntax", "--enforce-nullif", "--print-fixes"]
        if tables_dir and Path(tables_dir).exists():
            cmd += ["--tables-dir", str(tables_dir)]
        if patch:
            cmd += ["--out-sql", f_out.name]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        patched_sql = Path(f_out.name).read_text(encoding="utf-8") if patch else sql
        return {"log": stdout + "\n" + stderr, "sql": patched_sql}

# ---------- Prompt ----------
MINIMAL_RULES = """[Rules]
- Examples marked with * are most similar to the current question - follow their patterns and structure closely. Mimic their JOIN order, subquery approach (CTE vs NOT EXISTS vs LEFT JOIN), aggregation, and filtering logic.
- **MANDATORY**: Always include all JOIN key columns in the SELECT clause. Every ID or key used to JOIN tables must be present in the final result set.
- **ABSOLUTE LAW**: Use ONLY columns that are explicitly listed in [Database Schema]. NO EXCEPTIONS. Do not invent, assume, infer, or guess ANY column name that does not appear verbatim in [Database Schema]. A column that is not in the schema does NOT exist and will cause a query failure.
- **CRITICAL**: STRICTLY adhere to the [Database Schema] section. If a table or column is not in [Database Schema], DO NOT use it under any circumstances. Do NOT add, assume, or reference any tables or columns not provided in the schema. This is a hard constraint.
- **CRITICAL VALIDATION RULE**: BEFORE writing any SQL, scan the [Database Schema] and make a mental list of ALL available columns for each table. Then ONLY select from this list. If you try to use a column name you are not 100% certain is in the schema, you will cause a query failure. DO NOT GUESS.
- **CRITICAL**: SELECT ONLY NECESSARY COLUMNS. Do NOT select all columns from all tables. Select only: (1) Primary/foreign key columns for JOINs, (2) Columns in user's request, (3) Columns essential to answer the question. Be SELECTIVE and MINIMAL in every query.
- **MANDATORY**: BEFORE including any column in your SELECT, scan [Database Schema] and verify it is listed EXACTLY as shown. EVERY column name must match the exact spelling and case shown in [Database Schema]. If you are unsure about a column name, DO NOT use it. Accuracy is critical - a misspelled or non-existent column will cause total failure.
- **FORBIDDEN**: Do NOT hallucinate columns. Do NOT guess column names. Do NOT assume similar columns exist if not in schema. Do NOT try variations of column names. Do NOT use common column naming patterns unless they are explicitly in the schema. Use ONLY exact column names from [Database Schema].
- **IF NOT IN SCHEMA, IT DOES NOT EXIST**: The [Database Schema] shows EVERY column available for each table. Any column not listed is NOT available, was NEVER available, and will NEVER be available. Do NOT attempt to reference it, invoke it, or assume it exists. Attempting to use a non-existent column will cause the query to fail.
- **CRITICAL EXAMPLES TO AVOID**: Never use AMOUNT_RECEIVED, AMOUNT_REJECTED on PO_DISTRIBUTIONS_ALL (only AMOUNT_DELIVERED, AMOUNT_CANCELLED exist). Never use CUSTOMER_NAME if only CUSTOMER_ID is in schema. Never use non-existent column variations. Check schema TWICE before using any column. When in doubt, OMIT the column.
- **CRITICAL EXAMPLES TO AVOID**: Never use EXPENSE_ACCOUNT, TAX_CODE on AP_INVOICE_DISTRIBUTIONS_ALL_VAULT. Never use non-existent column variations. Check schema TWICE before using any column. When in doubt, OMIT the column.
- **SPELLING PRECISION**: ALWAYS use exact spellings from [Database Schema]. CRITICAL: NEVER use 'ACCTRUAL_ACCOUNT_ID' or 'ACCTUAL_ACCOUNT_ID' (both misspellings). The correct spelling is 'ACCRUAL_ACCOUNT_ID' (two C's: AC-CRUAL). Triple-check spelling for all columns.
- **DO NOT SELECT ACCRUAL_ACCOUNT_ID UNLESS REQUESTED**: Unless the user explicitly asks for "accrual account", "accounting", "accrual information", or "accrual details", DO NOT include ACCRUAL_ACCOUNT_ID in the SELECT clause. For general PO detail queries, omit this column.
- **CORRECT USAGE**: When selecting from PO_DISTRIBUTIONS_ALL, the correct accrual column is: pd.ACCRUAL_ACCOUNT_ID (exactly as shown, two C's before RUAL). Example: SELECT pd.ACCRUAL_ACCOUNT_ID, pd.AMOUNT_DELIVERED FROM PO_DISTRIBUTIONS_ALL pd
- **MINIMAL COLUMN SELECTION**: When no specific column list is provided, be EXTREMELY SELECTIVE. Select ONLY: table primary keys, foreign keys for JOINs, columns directly relevant to answer the question (e.g., for "receiving details" use receiving-related columns only, NOT all columns from all tables). NEVER select unnecessary accounting columns, dates, or flags unless explicitly requested.
- **ABSOLUTE CONSTRAINT**: The column list in [Database Schema] is EXHAUSTIVE and COMPLETE. Any column NOT shown in [Database Schema] does NOT exist in the database and cannot be queried. Do NOT attempt to use, reference, or guess columns beyond those explicitly listed. Every column name MUST appear in the provided schema or the query will fail.
- When computing closing balance, use this expression exactly:
  NVL(BEGIN_BALANCE_DR,0) - NVL(BEGIN_BALANCE_CR,0)
  + NVL(PERIOD_NET_DR,0) - NVL(PERIOD_NET_CR,0)
- Do NOT reference a column alias inside CASE unless it is computed in a subquery.
  EITHER inline the full expression inside CASE (preferred),
  OR compute it in a subquery and reference it in the outer SELECT.
- Always guard divisions:  / NULLIF(ABS(<denominator>), 0)
- Prefer PERIOD_NAME-based filters (e.g., 'Jan-03','Jun-03'), not literal START_DATEs.
- CRITICAL: Month abbreviations in PERIOD_NAME must ALWAYS be in title case format: 'Jan-03', 'Feb-03', 'Mar-03', 'Apr-03', 'May-03', 'Jun-03', 'Jul-03', 'Aug-03', 'Sep-03', 'Oct-03', 'Nov-03', 'Dec-03'. Do NOT use uppercase (MAR-03, APR-03) or lowercase (mar-03, apr-03).
- CRITICAL: Use proper table alias names in JOIN clauses and be STRICTLY CONSISTENT when referencing columns. For EVERY table in FROM/JOIN, assign ONE alias and use ONLY that alias for ALL column references. Example: if you write "JOIN AP_SUPPLIER_SITES_ALL ass ON ...", then EVERY reference to AP_SUPPLIER_SITES_ALL columns MUST use 'ass' (e.g., ass.VENDOR_SITE_ID). Do NOT use the wrong alias (e.g., aps.VENDOR_SITE_ID if aps refers to AP_SUPPLIERS). Verify each column reference matches its table's alias defined in the FROM/JOIN clauses.
- **MANDATORY**: When MTL_SYSTEM_ITEMS_B table is used, you MUST join on BOTH ORGANIZATION_ID AND INVENTORY_ITEM_ID columns. These two columns together constitute the primary key for MTL_SYSTEM_ITEMS_B. NEVER join on INVENTORY_ITEM_ID alone. The join condition must be: table_alias.ORGANIZATION_ID = source_table.ORG_ID AND table_alias.INVENTORY_ITEM_ID = source_table.ITEM_ID
- **MANDATORY**: When joining invoice data (AP_INVOICE_DISTRIBUTIONS_ALL) with purchasing data (PO_DISTRIBUTIONS_ALL), you MUST use the PO_HEADER_ID column: aid.PO_HEADER_ID = pd.PO_HEADER_ID. This is the primary linking key between invoices and purchase orders.
""".strip()

def compose_prompt(question: str, plan_dict: Dict[str, Any], cfg: APIConfig) -> Tuple[str, Dict[str, Any]]:
    logger.info(f"📌 compose_prompt called:")
    logger.info(f"   - question length: {len(question)} chars")
    
    tables_dir = Path(cfg.tables_dir)
    db_schema = build_db_schema(plan_dict, tables_dir)

    planner_fewshots = plan_dict.get("fewshots", [])
    prior = select_priority_corrections(question, plan_dict, Path(cfg.feedback_store), k=2)
    
    selected_examples = []
    
    top_semantic = planner_fewshots[:2] if len(planner_fewshots) >= 2 else planner_fewshots
    selected_examples.extend(top_semantic)
    
    planner_ids = {ex.get("id") for ex in planner_fewshots}
    for ex in prior:
        corr_id = ex.get("id", "corr")
        if corr_id not in planner_ids:
            selected_examples.append({
                "id": corr_id,
                "intent": ex.get("question", ""),
                "tables": ex.get("tables", []),
                "sql": ex.get("corrected_sql", ex.get("sql", ""))
            })
    
    added_ids = {ex.get("id") for ex in selected_examples}
    for ex in planner_fewshots[2:]:
        if ex.get("id") not in added_ids:
            selected_examples.append(ex)
            added_ids.add(ex.get("id"))
    
    logger.info(f"📋 FEWSHOT SELECTION SUMMARY (Total: {len(selected_examples)}, Will use: {min(len(selected_examples), cfg.fewshots_k)})")
    logger.info(f"   Planner provided: {len(planner_fewshots)} examples")
    logger.info(f"   Corrections found: {len(prior)} examples")
    logger.info(f"")
    logger.info(f"   Selected examples in order:")
    for idx, ex in enumerate(selected_examples[:cfg.fewshots_k], 1):
        ex_id = ex.get("id", "unknown")
        ex_question = ex.get("intent", "")[:100]
        logger.info(f"   [{idx}] ID={ex_id}: {ex_question}")
    
    if len(selected_examples) > cfg.fewshots_k:
        logger.info(f"")
        logger.info(f"   ⚠️  Dropped examples (exceeded fewshots_k={cfg.fewshots_k}):")
        for idx, ex in enumerate(selected_examples[cfg.fewshots_k:], cfg.fewshots_k + 1):
            ex_id = ex.get("id", "unknown")
            ex_question = ex.get("intent", "")[:100]
            logger.info(f"   [{idx}] ID={ex_id}: {ex_question}")

    examples_block = inject_examples_block(
        selected_examples,
        Path(cfg.fewshots),
        mode="sql",
        max_examples=cfg.fewshots_k,
        max_sql_chars=8000,
        user_question=question
    )

    guidance = ""
    if cfg.prompt_mode.lower() == "author7b" and author_prompt_with_7b:
        try:
            guidance = author_prompt_with_7b(question, plan_dict, cfg.model_id) or ""
        except Exception:
            guidance = ""

    reference_parts = []
    
    reference_parts.append(MINIMAL_RULES)
    
    has_very_similar = any("[* VERY SIMILAR]" in ex.get("intent", "") or "[VERY SIMILAR]" in ex.get("intent", "") for ex in selected_examples[:cfg.fewshots_k])
    if has_very_similar:
        reference_parts.append("[VERY SIMILAR EXAMPLE FOUND]\n⚠️ IMPORTANT: The first example(s) marked as [* VERY SIMILAR] or [VERY SIMILAR] are semantically equivalent to your question. Copy their SQL structure EXACTLY - use the same JOIN approach, aggregation method, and filtering logic. Do NOT create a different solution.")
    
    if guidance:
        reference_parts.append("[Guidance]\n" + guidance.strip())
    
    if examples_block:
        reference_parts.append(examples_block)
    
    logger.info(f"   📋 reference_parts summary (before joining):")
    for i, part in enumerate(reference_parts, 1):
        preview = part[:100].replace('\n', ' ')
        logger.info(f"      [{i}] {preview}...")
    
    reference_block = "\n\n".join([p for p in reference_parts if p]).strip()
    logger.info(f"   🔗 reference_block created (length: {len(reference_block)} chars)")

    logger.info(f"   🛠️  Building final prompt...")
    logger.info(f"      - DB Schema length: {len(db_schema)} chars")
    logger.info(f"      - Reference Block length: {len(reference_block)} chars")
    
    prompt = f"""You are a PostgreSQL expert. Generate executable SQL based on the user's question.
Only output SQL query. Do not invent columns - use only those in the schema.
CRITICAL: Columns marked with **[ESSENTIAL]** are mandatory for proper joins and aggregations - prioritize them.

[User Question]
{question}

[Database Schema]
{db_schema}

[Reference Information]
{reference_block}

[User Question]
{question}

```sql
"""
    
    logger.info(f"   ✅ Final prompt constructed (total length: {len(prompt)} chars)")
    logger.info(f"   📏 Final prompt length: {len(prompt)} chars")
    logger.info(f"   📌 Prompt structure verified and ready to send to LLM")
    
    logger.info(f"   📖 [Reference Information] section content:")
    ref_start = prompt.find("[Reference Information]")
    ref_end = prompt.find("[User Question]", ref_start)
    if ref_start > 0 and ref_end > 0:
        ref_section = prompt[ref_start:ref_end].strip()
        ref_lines = ref_section.split('\n')
        logger.info(f"---BEGIN REFERENCE SECTION---")
        for line in ref_lines[:30]:
            logger.info(line)
        if len(ref_lines) > 30:
            num_more = len(ref_lines) - 30
            logger.info(f"... ({num_more} more lines)")
        logger.info(f"---END REFERENCE SECTION---")
    
    try:
        prompt_log_path = Path(cfg.tables_dir).parent.parent / "logs" / "last_prompt.txt"
        prompt_log_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_log_path.write_text(prompt, encoding="utf-8")
        logger.info(f"💾 Full prompt saved to: {prompt_log_path}")
    except Exception as e:
        logger.warning(f"Failed to save prompt to file: {e}")
    
    return prompt, {"examples_used": [e.get("id") for e in selected_examples[:cfg.fewshots_k]]}

# ---------- API models ----------
class InferenceRequest(BaseModel):
    question: str
    filters: Optional[str] = None
    group_columns: Optional[str] = None
    columns_list: Optional[str] = None

class InferenceResponse(BaseModel):
    sql: str
    success: bool
    message: str
    plan_info: Dict[str, Any]

class CorrectionRequest(BaseModel):
    question: str
    candidate_sql: Optional[str] = None
    corrected_sql: str
    tables: Optional[List[str]] = None
    rationale: Optional[str] = None
    created_by: Optional[str] = None  # e.g., "Mohan"

class CorrectionResponse(BaseModel):
    ok: bool
    message: str

# ---------- App ----------
app = FastAPI(title="askGuru ARAG API (fixed)", version="0.4")

# Global model cache (loaded at startup)
_MODEL = None
_TOKENIZER = None

def load_finetuned_model(model_path: str):
    """
    Load XiYanSQL-QwenCoder-32B model and tokenizer from HuggingFace.
    Optimized for single A100 80GB GPU inference.
    Models are cached in HF_HOME (/askGuruARAG/.hf).
    
    Follows the approach from XiYanSQL_32b_example_usage.py:
    - Minimal parameters for reliable loading
    - device_map="auto" handles memory distribution
    """
    try:
        # Detect GPU and log specs
        num_gpus = torch.cuda.device_count()
        logger.info(f"📊 Detected {num_gpus} GPU(s)")
        if num_gpus > 0:
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        logger.info(f"Loading model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        logger.info(f"Loading tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model.eval()
        logger.info("✅ Model and tokenizer loaded successfully")
        logger.info(f"🚀 Model ready for inference")
        return tokenizer, model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def _cleanup_models(verbose: bool = True):
    """
    Aggressively clean up GPU memory from inference.
    This is called synchronously after each inference to prevent OOM.
    """
    global _MODEL, _TOKENIZER
    try:
        import gc
        import torch
        
        if verbose:
            logger.info("🧹 Starting aggressive memory cleanup...")
        
        # 1. Delete model and tokenizer references
        if _MODEL is not None:
            if verbose:
                logger.info("   - Deleting model from GPU...")
            del _MODEL
        if _TOKENIZER is not None:
            if verbose:
                logger.info("   - Deleting tokenizer...")
            del _TOKENIZER
        _MODEL, _TOKENIZER = None, None
        
        # 2. Force Python garbage collection multiple times
        if verbose:
            logger.info("   - Running Python garbage collection...")
        for _ in range(3):
            gc.collect()
        
        # 3. Clear CUDA cache aggressively
        if verbose:
            logger.info("   - Clearing CUDA cache...")
        torch.cuda.empty_cache()
        
        # 4. Try to clear CUDA cached memory
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
        if hasattr(torch.cuda, 'reset_accumulate_memory_stats'):
            torch.cuda.reset_accumulate_memory_stats()
        
        # 5. Log final GPU memory status across all GPUs
        if verbose and torch.cuda.is_available():
            logger.info(f"   ✅ Memory cleanup complete (All GPUs):")
            total_allocated = 0
            total_reserved = 0
            total_capacity = 0
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                total_allocated += mem_allocated
                total_reserved += mem_reserved
                total_capacity += mem_total
                logger.info(f"      - GPU {i}: {mem_allocated:.2f}/{mem_total:.2f} GiB allocated, {mem_reserved:.2f} GiB reserved")
            logger.info(f"      - Total: {total_allocated:.2f}/{total_capacity:.2f} GiB allocated across {torch.cuda.device_count()} GPUs")
        else:
            logger.info(f"   ✓ Memory cleanup complete")
            
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")

def _get_model(cfg: APIConfig):
    """Get the loaded model and tokenizer (reload if None)"""
    global _MODEL, _TOKENIZER
    
    # If caching is disabled, always load fresh model
    if not cfg.cache_model:
        logger.info("   ℹ️  Model caching is DISABLED - loading fresh model for each inference")
        _cleanup_models(verbose=False)
        _TOKENIZER, _MODEL = load_finetuned_model(cfg.model_id)
        return _TOKENIZER, _MODEL
    
    # If model is None but caching is enabled, reload it (safety fallback)
    if _MODEL is None or _TOKENIZER is None:
        logger.warning("⚠️  Model was unloaded but caching is enabled. Reloading model...")
        _TOKENIZER, _MODEL = load_finetuned_model(cfg.model_id)
        return _TOKENIZER, _MODEL
    
    return _TOKENIZER, _MODEL

def generate(mdl, tok, prompt: str, config: GenConfig):
    """
    Generate text using the finetuned model.
    This function replicates the behavior from llm_utils but uses the loaded model directly.
    """
    input_ids = tok(prompt, return_tensors="pt").to(mdl.device)
    gen = mdl.generate(
        **input_ids,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.temperature > 0,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    out = tok.decode(gen[0], skip_special_tokens=True)
    if out.startswith(prompt):
        out = out[len(prompt):]
    if config.stop:
        for s in config.stop:
            cut = out.find(s)
            if cut != -1:
                out = out[:cut]
                break
    return out.strip()

@app.on_event("startup")
async def startup_event():
    """Load finetuned model and tokenizer on startup"""
    global _MODEL, _TOKENIZER
    try:
        logger.info("=" * 80)
        logger.info("Starting up the FastAPI application...")
        logger.info(f"Model path: {MODEL_PATH}")
        logger.info("=" * 80)
        _TOKENIZER, _MODEL = load_finetuned_model(MODEL_PATH)
        logger.info("✓ Model and tokenizer loaded successfully!")
        logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"✓ Number of GPUs: {num_gpus}")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"  - GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"✗ Failed to load model on startup: {str(e)}")
        raise

@app.get("/health")
def health():
    gpu_info = {}
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_info = {
            "cuda_available": True,
            "num_gpus": num_gpus,
            "gpus": []
        }
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated(i) / 1024**3
            gpu_info["gpus"].append({
                "id": i,
                "name": gpu_name,
                "total_memory_gb": round(gpu_memory, 2),
                "allocated_memory_gb": round(gpu_allocated, 2)
            })
    else:
        gpu_info = {"cuda_available": False}
    
    return {
        "ok": True, 
        "config": CFG.__dict__,
        "model_loaded": _MODEL is not None and _TOKENIZER is not None,
        "gpu_info": gpu_info,
        "model_path": MODEL_PATH
    }

def build_enriched_question(question: str, filters: Optional[str] = None, group_columns: Optional[str] = None, columns_list: Optional[str] = None) -> str:
    """Build enriched question by concatenating all provided inputs with descriptive prefixes."""
    logger.info(f"[build_enriched_question] Building enriched question...")
    logger.info(f"  - Input question: {question[:300]}...")
    logger.info(f"  - Filters provided: {bool(filters and filters.strip())}")
    logger.info(f"  - Group columns provided: {bool(group_columns and group_columns.strip())}")
    logger.info(f"  - Columns list provided: {bool(columns_list and columns_list.strip())}")
    
    enriched = question.strip().replace('\n', ' ')
    logger.info(f"  - Base enriched (normalized): {enriched[:100]}...")
    
    if filters and filters.strip():
        filters_clean = filters.strip().replace(chr(10), ' ').replace(chr(13), ' ')
        enriched += f" Using these filters : {filters_clean}"
        logger.info(f"  ✓ Added filters: {filters_clean[:80]}...")
    
    if group_columns and group_columns.strip():
        group_clean = group_columns.strip().replace(chr(10), ' ').replace(chr(13), ' ')
        enriched += f". Group by these columns : {group_clean}"
        logger.info(f"  ✓ Added group columns: {group_clean[:80]}...")
    
    if columns_list and columns_list.strip():
        cols_clean = columns_list.strip().replace(chr(10), ' ').replace(chr(13), ' ')
        enriched += f". And must return these columns only in the final query : {cols_clean}"
        logger.info(f"  ✓ Added columns list: {cols_clean[:80]}...")
    
    enriched += " and also get all the primary keys from the tables used to generate the sql"
    logger.info(f"  ✓ Added primary key directive")
    
    logger.info(f"  ✅ Final enriched question: {enriched[:150]}...")
    
    return enriched

@app.post("/inference", response_model=InferenceResponse)
def inference(req: InferenceRequest) -> InferenceResponse:
    import logging
    import torch
    import gc
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=" * 80)
        logger.info("INFERENCE - START")
        logger.info("=" * 80)
        logger.info(f"⚙️  MEMORY CONFIG: Model caching = {CFG.cache_model}")
        
        # Pre-inference memory check and cleanup (all GPUs)
        logger.info("🔍 PRE-INFERENCE GPU MEMORY STATUS (All GPUs):")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            total_allocated = 0
            total_reserved = 0
            total_capacity = 0
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                total_allocated += mem_allocated
                total_reserved += mem_reserved
                total_capacity += mem_total
                logger.info(f"  - GPU {i}: {mem_allocated:.2f}/{mem_total:.2f} GiB allocated, {mem_reserved:.2f} GiB reserved")
            logger.info(f"  - Total: {total_allocated:.2f}/{total_capacity:.2f} GiB allocated across {torch.cuda.device_count()} GPUs")
            
            # If memory usage is too high, cleanup before inference
            if total_allocated > total_capacity * 0.7:
                logger.warning(f"  ⚠️  HIGH GPU memory usage ({total_allocated:.2f}/{total_capacity:.2f} GiB). Running cleanup before inference...")
                _cleanup_models(verbose=True)
                gc.collect()
                torch.cuda.empty_cache()
        
        logger.info(f"Request received:")
        logger.info(f"  - Question: {req.question}")
        logger.info(f"  - Filters: {req.filters if req.filters else 'None'}")
        logger.info(f"  - Group Columns: {req.group_columns if req.group_columns else 'None'}")
        logger.info(f"  - Columns List: {req.columns_list if req.columns_list else 'None'}")
        
        enriched_question = build_enriched_question(req.question, req.filters, req.group_columns, req.columns_list)
        logger.info(f"  📝 Enriched question: {enriched_question}")
        
        if not req.question or not req.question.strip():
            logger.warning("  ✗ Empty question received")
            raise HTTPException(400, "Empty question")

        # OOD Gate Check
        logger.info("-" * 80)
        logger.info("STEP 1: Out-of-Domain (OOD) Gate Check...")
        logger.info(f"  - Index directory: {CFG.index_dir}")
        logger.info(f"  - Tables directory: {CFG.tables_dir}")
        verdict = gate_or_message(enriched_question, Path(CFG.index_dir), Path(CFG.tables_dir), debug=False)
        logger.info(f"  ✓ OOD check complete")
        logger.info(f"  - In scope: {verdict.get('in_scope', True)}")
        if not verdict.get("in_scope", True):
            logger.info(f"  - Message: {verdict.get('message', 'Out of scope')}")
            logger.info("=" * 80)
            logger.info("INFERENCE - END (Out of scope)")
            logger.info("=" * 80)
            return InferenceResponse(sql="", success=False, message=verdict.get("message","Out of scope"), plan_info={"in_scope": False})

        # Planning Phase
        logger.info("-" * 80)
        logger.info("STEP 2: Planning Phase - Table and Column Selection...")
        logger.info(f"  - Loading table cards from: {CFG.tables_dir}")
        cards = load_cards(Path(CFG.tables_dir))
        logger.info(f"  ✓ Loaded {len(cards)} table cards")
        
        logger.info(f"  - Routing tables using keyword map: {CFG.router_map}")
        seeds = route_tables(enriched_question, set(cards.keys()), map_path=Path(CFG.router_map) if Path(CFG.router_map).exists() else None)
        logger.info(f"  ✓ Seed tables identified: {seeds}")
        
        logger.info(f"  - Running planner with:")
        logger.info(f"    - max_tables: {CFG.max_tables}")
        logger.info(f"    - per_table_cols: {CFG.per_table_cols}")
        logger.info(f"    - fewshots_k: {CFG.fewshots_k}")
        logger.info(f"    - index_dir: {CFG.index_dir}")
        logger.info(f"    - fewshots: {CFG.fewshots}")
        logger.info(f"    - joins_path: {CFG.joins}")
        logger.info(f"    - vectors_dir: {CFG.vectors_dir}")
        
        P = plan(
            enriched_question,
            Path(CFG.index_dir),
            Path(CFG.tables_dir),
            Path(CFG.fewshots),
            joins_path=Path(CFG.joins) if Path(CFG.joins).exists() else None,
            max_tables=CFG.max_tables,
            per_table_cols=CFG.per_table_cols,
            fewshots_k=CFG.fewshots_k,
            seed_tables=seeds,
            vectors_dir=Path(CFG.vectors_dir) if Path(CFG.vectors_dir).exists() else None
        )
        logger.info(f"  ✓ Planning complete")
        logger.info(f"  - In scope: {P.get('in_scope', True)}")
        logger.info(f"  - Tables selected: {P.get('tables_selected', [])}")
        logger.info(f"  - Join path: {P.get('join_path', [])}")
        logger.info(f"  - Join conditions: {P.get('join_conds', [])}")
        logger.info(f"  - Few-shot examples: {[ex.get('id') for ex in P.get('fewshots', [])]}")
        
        if not P.get("in_scope", True):
            logger.info(f"  - Message: {P.get('message', 'Out of scope')}")
            logger.info("=" * 80)
            logger.info("INFERENCE - END (Out of scope after planning)")
            logger.info("=" * 80)
            return InferenceResponse(sql="", success=False, message=P.get("message","Out of scope"), plan_info=P)

        # Prompt Composition
        logger.info("-" * 80)
        logger.info("STEP 3: Prompt Composition...")
        logger.info(f"  - Prompt mode: {CFG.prompt_mode}")
        logger.info(f"  - Feedback store: {CFG.feedback_store}")
        prompt, meta = compose_prompt(enriched_question, P, CFG)
        logger.info(f"  ✓ Prompt composed (length: {len(prompt)} chars)")
        logger.info(f"  - Examples used: {meta.get('examples_used', [])}")
        logger.info(f"  Complete prompt being sent to model:\n{'='*80}\n{prompt}\n{'='*80}")

        # Generation
        logger.info("-" * 80)
        logger.info("STEP 4: SQL Generation...")
        logger.info(f"  - Loading model: {CFG.model_id}")
        tok, mdl = _get_model(CFG)
        logger.info(f"  ✓ Model loaded")
        logger.info(f"  - Model device: {mdl.device}")
        logger.info(f"  Generation parameters:")
        logger.info(f"    - max_new_tokens: {CFG.max_new_tokens}")
        logger.info(f"    - temperature: {CFG.temperature}")
        logger.info(f"    - top_p: {CFG.top_p}")
        logger.info(f"    - repetition_penalty: {CFG.repetition_penalty}")
        
        out = generate(mdl, tok, prompt, GenConfig(
            model_id=CFG.model_id,
            max_new_tokens=CFG.max_new_tokens,
            temperature=CFG.temperature,
            top_p=CFG.top_p,
            repetition_penalty=CFG.repetition_penalty,
        ))
        logger.info(f"  ✓ Generation complete (output length: {len(out)} chars)")
        logger.info(f"  Raw output preview (first 500 chars):\n{out[:500]}...")
        
        raw_sql = extract_sql(out)
        logger.info(f"  ✓ SQL extracted (length: {len(raw_sql)} chars)")
        logger.info(f"  Raw SQL:\n{raw_sql}")
        
        # POST-GENERATION MEMORY MONITORING (All GPUs)
        logger.info("-" * 80)
        logger.info("📊 POST-GENERATION GPU MEMORY STATUS (All GPUs):")
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure all GPU operations are done
            total_allocated = 0
            total_reserved = 0
            total_capacity = 0
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                total_allocated += mem_allocated
                total_reserved += mem_reserved
                total_capacity += mem_total
                logger.info(f"  - GPU {i}: {mem_allocated:.2f}/{mem_total:.2f} GiB allocated, {mem_reserved:.2f} GiB reserved")
            logger.info(f"  - Total: {total_allocated:.2f}/{total_capacity:.2f} GiB allocated across {torch.cuda.device_count()} GPUs")
        
        # Clear output tensors early
        del out
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("  ✓ Cleared generation output from memory")

        # SQL Guard
        logger.info("-" * 80)
        logger.info("STEP 5: SQL Guard - Validation and Patching...")
        policy = (CFG.guard_policy or "patch-only").lower()
        logger.info(f"  - Guard policy: {policy}")
        final_sql = raw_sql
        guard_log = ""
        
        if policy == "detect-only":
            logger.info(f"  - Running guard in detect-only mode...")
            g = run_sql_guard(raw_sql, Path(CFG.tables_dir), patch=False)
            guard_log = g["log"]
            logger.info(f"  ✓ Guard check complete (no patching)")
            logger.info(f"  Guard log:\n{guard_log}")
        elif policy == "patch-only":
            logger.info(f"  - Running guard in patch-only mode...")
            g = run_sql_guard(raw_sql, Path(CFG.tables_dir), patch=True)
            final_sql = g["sql"]
            guard_log = g["log"]
            logger.info(f"  ✓ Guard check and patching complete")
            logger.info(f"  Guard log:\n{guard_log}")
            if final_sql != raw_sql:
                logger.info(f"  ⚠ SQL was modified by guard")
                logger.info(f"  Patched SQL:\n{final_sql}")
            else:
                logger.info(f"  ✓ No modifications needed")
        else:  # retry-then-patch
            logger.info(f"  - Running guard in retry-then-patch mode...")
            logger.info(f"  - Initial guard check...")
            g1 = run_sql_guard(raw_sql, Path(CFG.tables_dir), patch=False)
            logger.info(f"  ✓ Initial guard check complete")
            logger.info(f"  Initial guard log:\n{g1['log']}")
            
            if "NULLIF" in g1["log"] or "Per-month CTE names" in g1["log"] or "Unknown columns" in g1["log"]:
                logger.info(f"  ⚠ Issues detected, retrying with stricter rules...")
                strict = MINIMAL_RULES + "\n- Re-check division guards and avoid per-month CTEs."
                prompt2 = prompt.replace(MINIMAL_RULES, strict)
                logger.info(f"  - Regenerating with stricter prompt...")
                out2 = generate(mdl, tok, prompt2, GenConfig(
                    model_id=CFG.model_id,
                    max_new_tokens=CFG.max_new_tokens,
                    temperature=CFG.temperature,
                    top_p=CFG.top_p,
                    repetition_penalty=CFG.repetition_penalty,
                ))
                logger.info(f"  ✓ Retry generation complete")
                raw_sql2 = extract_sql(out2)
                logger.info(f"  Retry SQL:\n{raw_sql2}")
                logger.info(f"  - Running guard on retry SQL...")
                g2 = run_sql_guard(raw_sql2, Path(CFG.tables_dir), patch=True)
                final_sql = g2["sql"]
                guard_log = g1["log"] + "\n---RETRY---\n" + g2["log"]
                logger.info(f"  ✓ Retry guard check complete")
                logger.info(f"  Retry guard log:\n{g2['log']}")
            else:
                logger.info(f"  ✓ No major issues detected, applying patch...")
                g = run_sql_guard(raw_sql, Path(CFG.tables_dir), patch=True)
                final_sql = g["sql"]
                guard_log = g["log"]
                logger.info(f"  ✓ Patching complete")

        # Final Response
        logger.info("-" * 80)
        logger.info("STEP 6: Preparing Final Response...")
        plan_info = {
            "in_scope": True,
            "tables": P.get("tables_selected", []),
            "join_path": " -> ".join(P.get("join_path", [])),
            "join_conds": P.get("join_conds", []),
            "examples_used": meta.get("examples_used", []),
            "guard": guard_log[:2000]
        }
        logger.info(f"  ✓ Response prepared")
        logger.info(f"  Final SQL (length: {len(final_sql)} chars):")
        logger.info(f"\n{final_sql}\n")
        logger.info(f"  Plan info:")
        logger.info(f"    - Tables: {plan_info['tables']}")
        logger.info(f"    - Join path: {plan_info['join_path']}")
        logger.info(f"    - Examples used: {plan_info['examples_used']}")
        logger.info("=" * 80)
        logger.info("INFERENCE - END (Success)")
        logger.info("=" * 80)
        
        response = InferenceResponse(sql=final_sql.rstrip(';'), success=True, message="ok", plan_info=plan_info)
        
        # Log the final JSON response being sent
        logger.info("-" * 80)
        logger.info("FINAL JSON RESPONSE:")
        logger.info("-" * 80)
        response_dict = response.model_dump()
        logger.info(json.dumps(response_dict, indent=2, ensure_ascii=False))
        logger.info("-" * 80)
        
        # Cleanup embedder to free GPU memory (always, regardless of cache_model setting)
        # Embedders are lightweight and always lazy-loaded, so always cleanup after inference
        cleanup_embedder()
        
        # SYNCHRONOUS cleanup - only if model caching is disabled
        if not CFG.cache_model:
            logger.info("-" * 80)
            logger.info("🧹 Cleanup after inference (caching disabled)")
            _cleanup_models(verbose=True)
            logger.info("-" * 80)
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is, but cleanup embedder and models
        try:
            cleanup_embedder()
        except Exception as e:
            logger.warning(f"Embedder cleanup failed: {e}")
        
        if not CFG.cache_model:
            logger.warning("Cleaning up after HTTPException (caching disabled)...")
            _cleanup_models(verbose=False)
        raise
    except Exception as e:
        logger.error("=" * 80)
        logger.error("INFERENCE - ERROR")
        logger.error("=" * 80)
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception message: {str(e)}")
        logger.error(f"Exception details:", exc_info=True)
        logger.error("=" * 80)
        
        # Cleanup embedder after error
        try:
            cleanup_embedder()
        except Exception as cleanup_err:
            logger.warning(f"Embedder cleanup after error failed: {cleanup_err}")
        
        # SYNCHRONOUS cleanup - only if model caching is disabled
        if not CFG.cache_model:
            try:
                logger.error("Performing emergency cleanup after error (caching disabled)...")
                _cleanup_models(verbose=True)
            except Exception as cleanup_err:
                logger.error(f"Cleanup after error also failed: {cleanup_err}")
        
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

# ---------- Feedback endpoints ----------
@app.post("/feedback/correction", response_model=CorrectionResponse)
def submit_correction(req: CorrectionRequest) -> CorrectionResponse:
    if not req.corrected_sql or not req.question:
        raise HTTPException(400, "Both 'question' and 'corrected_sql' are required")
    entry = {
        "id": int(time.time() * 1000),
        "question": req.question.strip(),
        "candidate_sql": (req.candidate_sql or "").strip(),
        "corrected_sql": req.corrected_sql.strip(),
        "tables": req.tables or parse_tables_from_sql(req.corrected_sql),
        "rationale": (req.rationale or "").strip(),
        "created_by": (req.created_by or "user").strip(),
        "ts": int(time.time())
    }
    append_jsonl(Path(CFG.feedback_store), entry)
    return CorrectionResponse(ok=True, message="Saved correction. It will be used as a top-priority few-shot for similar questions.")

@app.get("/feedback/list")
def list_corrections(limit: int = 20):
    items = load_jsonl(Path(CFG.feedback_store))
    items = sorted(items, key=lambda x: x.get("ts", 0), reverse=True)[:limit]
    return {"count": len(items), "items": items}

@app.get("/config")
def get_config():
    return CFG.__dict__

# ================= Validation Query Extraction =================
from pydantic import BaseModel
from typing import Optional, List, Set

class ValidationQueryItem(BaseModel):
    original_parameters: Optional[str] = None
    parameter: Optional[str] = None
    description: Optional[str] = None
    purpose: Optional[str] = None
    sql: str

class ValidationQueryRequest(BaseModel):
    generated_sql: str
    original_question: str
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = False

class ValidationQueryResponse(BaseModel):
    validation_queries: List[ValidationQueryItem]
    success: bool
    message: str

def _tables_from_sql(sql: str) -> List[str]:
    # Extract table names mentioned in FROM/JOIN clauses
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"  [_tables_from_sql] Extracting tables from SQL...")
    logger.info(f"  [_tables_from_sql] SQL length: {len(sql)} chars")
    
    toks = re.findall(r'\bfrom\s+([a-zA-Z0-9_#\$]+)|\bjoin\s+([a-zA-Z0-9_#\$]+)',
                      sql, flags=re.IGNORECASE)
    logger.info(f"  [_tables_from_sql] Found {len(toks)} FROM/JOIN matches")
    
    tabs: Set[str] = set()
    for idx, (a, b) in enumerate(toks):
        t = (a or b)
        if t:
            tabs.add(t.upper())
            logger.info(f"  [_tables_from_sql]   Match {idx+1}: {t.upper()}")
    
    result = sorted(tabs)
    logger.info(f"  [_tables_from_sql] Final table list: {result}")
    return result


def _extract_where_clause(sql: str) -> str:
    """
    Extract WHERE clause(s) from SQL for focused analysis.
    Handles UNION queries by extracting all WHERE clauses.
    """
    if not sql:
        return ""
    
    # For UNION queries, extract all WHERE clauses
    if re.search(r'\bUNION\b', sql, re.IGNORECASE):
        where_clauses = []
        # Split by UNION and extract WHERE from each part
        parts = re.split(r'\bUNION\s+ALL\b|\bUNION\b', sql, flags=re.IGNORECASE)
        for part in parts:
            match = re.search(
                r'\bWHERE\b(.+?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|\bFETCH\b|$)', 
                part, 
                re.IGNORECASE | re.DOTALL
            )
            if match:
                where_clauses.append(match.group(1).strip())
        
        if where_clauses:
            # Combine all WHERE clauses, limit total length
            combined = "\n--- AND ---\n".join(where_clauses)
            if len(combined) > 500:
                combined = combined[:500] + "\n... (truncated)"
            return combined
    else:
        # Single query: extract WHERE clause
        match = re.search(
            r'\bWHERE\b(.+?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|\bFETCH\b|\bUNION\b|$)', 
            sql, 
            re.IGNORECASE | re.DOTALL
        )
        
        if match:
            where_clause = match.group(1).strip()
            # Limit to reasonable length for prompt
            if len(where_clause) > 500:
                where_clause = where_clause[:500] + "\n... (truncated)"
            return where_clause
    
    return ""


def _identify_module_from_tables(tables: List[str]) -> str:
    """
    Identify which module(s) the query belongs to based on tables used.
    Helps provide better context in the prompt.
    """
    tables_upper = [t.upper() for t in tables]
    
    modules = []
    
    # GL indicators
    gl_tables = {'GL_LEDGERS', 'GL_PERIODS', 'GL_BALANCES', 'GL_CODE_COMBINATIONS', 
                 'GL_JE_HEADERS', 'GL_JE_LINES', 'GL_ACCOUNTS'}
    if any(t in gl_tables for t in tables_upper):
        modules.append("GL")
    
    # Purchasing indicators
    po_tables = {'PO_HEADERS_ALL', 'PO_LINES_ALL', 'PO_VENDORS', 
                 'PO_LINE_LOCATIONS_ALL', 'PO_DISTRIBUTIONS_ALL', 'PO_AGENTS'}
    if any(t in po_tables for t in tables_upper):
        modules.append("Purchasing")
    
    # Payables indicators
    ap_tables = {'AP_INVOICES_ALL', 'AP_INVOICE_LINES_ALL', 'AP_SUPPLIERS',
                 'AP_PAYMENT_SCHEDULES_ALL', 'AP_INVOICE_DISTRIBUTIONS_ALL', 'AP_TERMS'}
    if any(t in ap_tables for t in tables_upper):
        modules.append("Payables")
    
    return ", ".join(modules) if modules else "Unknown"


def _extract_user_parameters_from_question(question: str) -> Dict[str, Any]:
    """
    Pre-extract potential user parameters from the question.
    This provides hints to the LLM and can be used for fallback.
    """
    if not question:
        return {}
    
    hints = {
        "quoted_phrases": re.findall(r'"([^"]+)"', question),
        "vendor_mentions": re.findall(r'\bvendor\s+([A-Z][A-Za-z\s&,\.]+?)(?:\s+(?:in|for|from|and|or|with|$))', question, re.IGNORECASE),
        "supplier_mentions": re.findall(r'\bsupplier\s+([A-Z][A-Za-z\s&,\.]+?)(?:\s+(?:in|for|from|and|or|with|$))', question, re.IGNORECASE),
        "ledger_mentions": re.findall(r'\bledger\s+([A-Z][A-Za-z\s\(\)]+?)(?:\s+(?:in|for|from|and|or|$))', question, re.IGNORECASE),
        "period_mentions": re.findall(r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b', question, re.IGNORECASE),
        "po_numbers": re.findall(r'\bPO[#\s]*(\d+)', question, re.IGNORECASE),
        # Fixed: Only match invoice followed by number/code (not just "invoices")
        # Must have at least one digit or be a proper code format
        "invoice_numbers": re.findall(r'\binvoice\s+(?:number\s+|#\s*)?([A-Z0-9][A-Z0-9-]{2,})', question, re.IGNORECASE),
        "operating_unit": re.findall(r'\boperating\s+unit\s+([A-Z][A-Za-z\s]+?)(?:\s+(?:in|for|from|and|or|$))', question, re.IGNORECASE),
    }
    
    return hints

def _generate_smart_fallback_queries(
    question: str,
    sql: str,
    tables: List[str],
    module: str,
    hints: Dict[str, Any],
    ledger_like: str = None,
    period_hits: List[tuple] = None
) -> List[ValidationQueryItem]:
    """
    Smart fallback that generates validation queries based on:
    1. Detected module (GL/Purchasing/Payables)
    2. Tables used in the query
    3. Hints extracted from the question
    """
    import logging
    logger = logging.getLogger(__name__)
    
    items = []
    tables_upper = [t.upper() for t in tables]
    
    logger.info(f"🔧 Smart fallback for module: {module}")
    
    def _phrase_to_like_token_local(phrase: str) -> str:
        """Convert phrase to LIKE token"""
        import re
        cleaned = re.sub(r"[^A-Za-z0-9]+", " ", phrase or "").strip()
        if not cleaned:
            return ""
        return "%".join(w for w in cleaned.split() if w)
    
    # GL Module fallbacks
    if "GL" in module:
        # Ledger validation
        if "GL_LEDGERS" in tables_upper and ledger_like:
            items.append(ValidationQueryItem(
                original_parameters=f'"{ledger_like.replace("%", " ")}"',
                parameter="ledger_name",
                description="Ledger name from SQL filter",
                purpose="Confirm intended ledger",
                sql=f"SELECT DISTINCT NAME\nFROM GL_LEDGERS\nWHERE LOWER(NAME) LIKE LOWER('%{ledger_like}%')"
            ))
            logger.info(f"  ✅ Added GL ledger validation: {ledger_like}")
        
        # Period validation
        if "GL_PERIODS" in tables_upper and period_hits:
            for (MON, YY) in period_hits:
                items.append(ValidationQueryItem(
                    original_parameters=f'"{MON} {YY}"',
                    parameter="period",
                    description="Accounting period",
                    purpose="Confirm intended period",
                    sql=f"SELECT DISTINCT PERIOD_NAME\nFROM GL_PERIODS\nWHERE LOWER(PERIOD_NAME) LIKE LOWER('%{MON}%{YY}%')"
                ))
                logger.info(f"  ✅ Added GL period validation: {MON} {YY}")
    
    # Purchasing Module fallbacks
    if "Purchasing" in module:
        # Vendor validation
        if "PO_VENDORS" in tables_upper:
            vendors = hints.get("vendor_mentions", [])
            for vendor in vendors[:2]:  # Max 2
                token = _phrase_to_like_token_local(vendor)
                if token:
                    items.append(ValidationQueryItem(
                        original_parameters=f'"{vendor}"',
                        parameter="vendor_name",
                        description="Vendor name from user input",
                        purpose="Confirm intended vendor",
                        sql=f"SELECT DISTINCT VENDOR_NAME\nFROM PO_VENDORS\nWHERE LOWER(VENDOR_NAME) LIKE LOWER('%{token}%')"
                    ))
                    logger.info(f"  ✅ Added PO vendor validation: {vendor}")
        
        # PO Number validation
        if "PO_HEADERS_ALL" in tables_upper:
            po_numbers = hints.get("po_numbers", [])
            for po_num in po_numbers[:2]:
                items.append(ValidationQueryItem(
                    original_parameters=f'"PO {po_num}"',
                    parameter="po_number",
                    description="Purchase Order number",
                    purpose="Confirm intended PO",
                    sql=f"SELECT DISTINCT SEGMENT1\nFROM PO_HEADERS_ALL\nWHERE SEGMENT1 LIKE '%{po_num}%'"
                ))
                logger.info(f"  ✅ Added PO number validation: {po_num}")
    
    # Payables Module fallbacks
    if "Payables" in module:
        # Supplier validation
        if "AP_SUPPLIERS" in tables_upper:
            suppliers = hints.get("supplier_mentions", []) or hints.get("vendor_mentions", [])
            for supplier in suppliers[:2]:
                token = _phrase_to_like_token_local(supplier)
                if token:
                    items.append(ValidationQueryItem(
                        original_parameters=f'"{supplier}"',
                        parameter="supplier_name",
                        description="Supplier name from user input",
                        purpose="Confirm intended supplier",
                        sql=f"SELECT DISTINCT VENDOR_NAME\nFROM AP_SUPPLIERS\nWHERE LOWER(VENDOR_NAME) LIKE LOWER('%{token}%')"
                    ))
                    logger.info(f"  ✅ Added AP supplier validation: {supplier}")
        
        # Invoice Number validation
        if "AP_INVOICES_ALL" in tables_upper:
            invoice_numbers = hints.get("invoice_numbers", [])
            for inv_num in invoice_numbers[:2]:
                items.append(ValidationQueryItem(
                    original_parameters=f'"Invoice {inv_num}"',
                    parameter="invoice_number",
                    description="Invoice number",
                    purpose="Confirm intended invoice",
                    sql=f"SELECT DISTINCT INVOICE_NUM\nFROM AP_INVOICES_ALL\nWHERE INVOICE_NUM LIKE '%{inv_num}%'"
                ))
                logger.info(f"  ✅ Added AP invoice validation: {inv_num}")
    
    # Common: Operating Unit (for both Purchasing and Payables)
    if ("Purchasing" in module or "Payables" in module) and "HR_OPERATING_UNITS" in tables_upper:
        operating_units = hints.get("operating_unit", [])
        for ou in operating_units[:1]:  # Max 1
            token = _phrase_to_like_token_local(ou)
            if token:
                items.append(ValidationQueryItem(
                    original_parameters=f'"{ou}"',
                    parameter="operating_unit",
                    description="Operating unit from user input",
                    purpose="Confirm intended operating unit",
                    sql=f"SELECT DISTINCT NAME\nFROM HR_OPERATING_UNITS\nWHERE LOWER(NAME) LIKE LOWER('%{token}%')"
                ))
                logger.info(f"  ✅ Added operating unit validation: {ou}")
    
    logger.info(f"🔧 Smart fallback generated {len(items)} validation queries")
    return items


def _schema_block_for_tables(tables: List[str], tables_dir: Path, max_cols_per_table: int = 12) -> str:
    """Compose a compact [Database Schema] block for the given tables using table cards."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"  [_schema_block_for_tables] Building schema for {len(tables)} tables")
    logger.info(f"  [_schema_block_for_tables] Tables directory: {tables_dir}")
    logger.info(f"  [_schema_block_for_tables] Max columns per table: {max_cols_per_table}")
    
    cards = load_cards(tables_dir)
    logger.info(f"  [_schema_block_for_tables] Loaded {len(cards)} table cards")
    
    lines: List[str] = []
    for T in tables:
        logger.info(f"  [_schema_block_for_tables] Processing table: {T}")
        card = cards.get(T, {})
        if not card:
            logger.warning(f"  [_schema_block_for_tables]   WARNING: No card found for table {T}")
        else:
            logger.info(f"  [_schema_block_for_tables]   Card found with {len(card.get('columns', []))} columns")
        
        essential_cols = card.get("essential_columns", [])
        if essential_cols:
            logger.info(f"  [_schema_block_for_tables]   Found {len(essential_cols)} essential columns")
        
        lines.append(f"{T} (")
        
        cols_added = 0
        essential_added = 0
        
        for ec in essential_cols:
            if isinstance(ec, dict):
                name = str(ec.get("name", "")).upper()
                reason = str(ec.get("reason", "")).strip()
                if name:
                    marker = "**[ESSENTIAL]**"
                    if reason:
                        lines.append(f"  {marker} {name} -- {reason}")
                    else:
                        lines.append(f"  {marker} {name}")
                    essential_added += 1
                    cols_added += 1
        
        if essential_added > 0:
            logger.info(f"  [_schema_block_for_tables]   Added {essential_added} essential columns for {T}")
        
        cols = card.get("columns", [])
        remaining_slot = max_cols_per_table - essential_added
        
        for c in cols[:remaining_slot]:
            name = str(c.get("name", "")).upper()
            ctype = str(c.get("type", "")).upper()
            desc = str(c.get("desc") or c.get("description") or "").strip()
            if name:
                if desc:
                    lines.append(f"  {name} {ctype} -- {desc}")
                else:
                    lines.append(f"  {name} {ctype}")
                cols_added += 1
        
        logger.info(f"  [_schema_block_for_tables]   Added {cols_added} total columns ({essential_added} essential) for {T}")
        lines.append(")\n")
    
    result = "\n".join(lines).strip()
    logger.info(f"  [_schema_block_for_tables] Schema block complete (length: {len(result)} chars)")
    return result

def parse_validation_queries(raw: str) -> List[ValidationQueryItem]:
    """
    Parse LLM response formatted with comment headers and ```sql fences.
    Expected pattern:
      -- Original Parameters: [...]
      -- Parameter: [...]
      -- Description: [...]
      -- Purpose: [...]
      ```sql
      SELECT ...
      ```
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("  [PARSER] Starting to parse validation queries...")
    logger.info(f"  [PARSER] Raw response length: {len(raw)} chars")
    logger.info(f"  [PARSER] Raw response:\n{raw}")
    
    lines = [l.rstrip() for l in raw.splitlines()]
    logger.info(f"  [PARSER] Split into {len(lines)} lines")
    
    items: List[ValidationQueryItem] = []
    cur = {"original_parameters": None, "parameter": None,
           "description": None, "purpose": None, "sql": ""}
    i = 0

    def flush():
        if cur.get("sql"):
            logger.info(f"  [PARSER] Flushing item:")
            logger.info(f"  [PARSER]   - original_parameters: {cur.get('original_parameters')}")
            logger.info(f"  [PARSER]   - parameter: {cur.get('parameter')}")
            logger.info(f"  [PARSER]   - description: {cur.get('description')}")
            logger.info(f"  [PARSER]   - purpose: {cur.get('purpose')}")
            logger.info(f"  [PARSER]   - sql: {cur.get('sql')[:100]}...")
            items.append(ValidationQueryItem(
                original_parameters=cur.get("original_parameters"),
                parameter=cur.get("parameter"),
                description=cur.get("description"),
                purpose=cur.get("purpose"),
                sql=cur.get("sql").strip().strip(';')
            ))
        else:
            logger.info(f"  [PARSER] Skipping flush - no SQL found in current item")

    while i < len(lines):
        L = lines[i]
        low = L.lower().strip()
        
        if low.startswith("-- original parameters:"):
            logger.info(f"  [PARSER] Line {i}: Found 'Original Parameters' header")
            if cur.get("sql"):
                flush()
                cur = {"original_parameters": None, "parameter": None,
                       "description": None, "purpose": None, "sql": ""}
            cur["original_parameters"] = L.split(":", 1)[1].strip() if ":" in L else None
            logger.info(f"  [PARSER]   Set original_parameters = {cur['original_parameters']}")
            
        elif low.startswith("-- parameter:"):
            logger.info(f"  [PARSER] Line {i}: Found 'Parameter' header")
            cur["parameter"] = L.split(":", 1)[1].strip() if ":" in L else None
            logger.info(f"  [PARSER]   Set parameter = {cur['parameter']}")
            
        elif low.startswith("-- description:"):
            logger.info(f"  [PARSER] Line {i}: Found 'Description' header")
            cur["description"] = L.split(":", 1)[1].strip() if ":" in L else None
            logger.info(f"  [PARSER]   Set description = {cur['description']}")
            
        elif low.startswith("-- purpose:"):
            logger.info(f"  [PARSER] Line {i}: Found 'Purpose' header")
            cur["purpose"] = L.split(":", 1)[1].strip() if ":" in L else None
            logger.info(f"  [PARSER]   Set purpose = {cur['purpose']}")
            
        elif low.startswith("```sql"):
            logger.info(f"  [PARSER] Line {i}: Found SQL code fence start")
            # capture until closing ```
            i += 1
            buf = []
            start_line = i
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            cur["sql"] = "\n".join(buf).strip()
            logger.info(f"  [PARSER]   Captured SQL from lines {start_line} to {i} ({len(buf)} lines)")
            logger.info(f"  [PARSER]   SQL content: {cur['sql'][:150]}...")
            if i < len(lines):
                logger.info(f"  [PARSER] Line {i}: Found SQL code fence end")
        
        i += 1
    
    logger.info(f"  [PARSER] Finished parsing loop, checking final item...")
    if cur.get("sql"):
        flush()
    
    logger.info(f"  [PARSER] Total items parsed: {len(items)}")
    logger.info(f"  [PARSER] Items with SQL: {len([it for it in items if it.sql])}")
    
    # If no items parsed, try alternative single-line format
    if not items:
        logger.info(f"  [PARSER] No items found with multi-line format, trying single-line format...")
        items = _parse_single_line_format(raw)
        logger.info(f"  [PARSER] Single-line parser found {len(items)} items")
    
    return [it for it in items if it.sql]

def _parse_single_line_format(raw: str) -> List[ValidationQueryItem]:
    """
    Fallback parser for single-line format like:
    -- Original Parameter: 'Jan-03' -- Parameter: Period Name -- Description: Validates...
    ```sql
    SELECT ...
    ```
    """
    import logging
    import re
    logger = logging.getLogger(__name__)
    
    logger.info("  [SINGLE-LINE PARSER] Attempting to parse single-line format...")
    
    items: List[ValidationQueryItem] = []
    
    # Split by ```sql blocks
    sql_blocks = re.split(r'```sql\s*', raw)
    logger.info(f"  [SINGLE-LINE PARSER] Found {len(sql_blocks)-1} SQL blocks")
    
    for i in range(1, len(sql_blocks)):
        block = sql_blocks[i]
        
        # Extract SQL (everything before closing ```)
        sql_match = re.search(r'^(.*?)```', block, re.DOTALL)
        if not sql_match:
            logger.info(f"  [SINGLE-LINE PARSER] Block {i}: No closing ``` found, skipping")
            continue
        
        sql = sql_match.group(1).strip()
        
        # Look for metadata in the previous block (before ```sql)
        metadata_block = sql_blocks[i-1]
        
        # Extract fields from single line or multiple lines
        original_param = None
        parameter = None
        description = None
        purpose = None
        
        # Try to find "-- Original Parameter:" or "-- Original Parameters:"
        orig_match = re.search(r'--\s*Original\s+Parameters?:\s*([^\-\n]+?)(?=\s*--|$)', metadata_block, re.IGNORECASE)
        if orig_match:
            original_param = orig_match.group(1).strip().strip("'\"")
        
        # Try to find "-- Parameter:"
        param_match = re.search(r'--\s*Parameter:\s*([^\-\n]+?)(?=\s*--|$)', metadata_block, re.IGNORECASE)
        if param_match:
            parameter = param_match.group(1).strip()
        
        # Try to find "-- Description:"
        desc_match = re.search(r'--\s*Description:\s*([^\-\n]+?)(?=\s*--|$)', metadata_block, re.IGNORECASE)
        if desc_match:
            description = desc_match.group(1).strip()
        
        # Try to find "-- Purpose:"
        purpose_match = re.search(r'--\s*Purpose:\s*([^\-\n]+?)(?=\s*--|$)', metadata_block, re.IGNORECASE)
        if purpose_match:
            purpose = purpose_match.group(1).strip()
        
        if sql:
            logger.info(f"  [SINGLE-LINE PARSER] Block {i}: Found query")
            logger.info(f"  [SINGLE-LINE PARSER]   - Original Parameters: {original_param}")
            logger.info(f"  [SINGLE-LINE PARSER]   - Parameter: {parameter}")
            logger.info(f"  [SINGLE-LINE PARSER]   - Description: {description}")
            logger.info(f"  [SINGLE-LINE PARSER]   - Purpose: {purpose}")
            logger.info(f"  [SINGLE-LINE PARSER]   - SQL: {sql[:100]}...")
            
            items.append(ValidationQueryItem(
                original_parameters=original_param,
                parameter=parameter,
                description=description,
                purpose=purpose,
                sql=sql.strip().strip(';')
            ))
    
    logger.info(f"  [SINGLE-LINE PARSER] Total items extracted: {len(items)}")
    return items

def _fallback_validation(question: str, generated_sql: str) -> List[ValidationQueryItem]:
    """
    Rule-of-thumb fallback: extract quoted strings or capitalized phrases from the question
    and generate simple validation queries for them.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("  [FALLBACK] Attempting rule-based fallback validation...")
    
    # Extract potential parameters from question
    # Look for quoted strings, dates, or capitalized phrases
    import re
    
    # Pattern 1: Quoted strings
    quoted = re.findall(r'["\']([^"\']+)["\']', question)
    
    # Pattern 2: Date-like patterns (Jan 07, January 2007, etc.)
    dates = re.findall(r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*[\-\s]*\d{2,4})\b', question, re.IGNORECASE)
    
    # Pattern 3: Capitalized phrases (2+ words starting with capitals)
    capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', question)
    
    params = list(set(quoted + dates + capitalized))
    logger.info(f"  [FALLBACK] Found {len(params)} potential parameters: {params}")
    
    if not params:
        logger.info("  [FALLBACK] No parameters found in question")
        return []
    
    # Extract tables from generated SQL
    tables = _tables_from_sql(generated_sql)
    logger.info(f"  [FALLBACK] Using tables from SQL: {tables}")
    
    items = []
    for param in params[:3]:  # Limit to 3 parameters
        # Try to determine if it's a date or a name
        is_date = bool(re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', param, re.IGNORECASE))
        
        if is_date:
            # Generate period validation query
            param_upper = param.upper()
            # Extract month and year parts
            month_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', param, re.IGNORECASE)
            year_match = re.search(r'\d{2,4}', param)
            
            if month_match and year_match:
                month = month_match.group(1).upper()[:3]
                year = year_match.group(0)
                if len(year) == 4:
                    year = year[-2:]  # Convert YYYY to YY
                
                sql = f"SELECT DISTINCT PERIOD_NAME FROM GL_PERIODS WHERE UPPER(PERIOD_NAME) LIKE '%{month}%{year}%'"
                
                items.append(ValidationQueryItem(
                    original_parameters=param,
                    parameter="Period Name",
                    description=f"Period validation for '{param}'",
                    purpose="Validate the period parameter entered by user",
                    sql=sql
                ))
                logger.info(f"  [FALLBACK] Created period validation for: {param}")
        else:
            # Generate name/ledger validation query
            # Use the first table that might contain names
            table_to_use = None
            for t in tables:
                if any(keyword in t.upper() for keyword in ['LEDGER', 'ORG', 'ENTITY', 'COMPANY']):
                    table_to_use = t
                    break
            
            if not table_to_use and tables:
                table_to_use = tables[0]
            
            if table_to_use:
                # Assume NAME column exists
                param_parts = param.upper().split()
                like_pattern = '%'.join(param_parts)
                
                sql = f"SELECT DISTINCT NAME FROM {table_to_use} WHERE UPPER(NAME) LIKE '%{like_pattern}%'"
                
                items.append(ValidationQueryItem(
                    original_parameters=param,
                    parameter="Name",
                    description=f"Name validation for '{param}'",
                    purpose="Validate the name parameter entered by user",
                    sql=sql
                ))
                logger.info(f"  [FALLBACK] Created name validation for: {param}")
    
    logger.info(f"  [FALLBACK] Generated {len(items)} fallback validation queries")
    return items


@app.post("/extract_validation_queries", response_model=ValidationQueryResponse)
async def extract_validation_queries(request: ValidationQueryRequest) -> ValidationQueryResponse:
    """
    Extract validation queries for user-provided parameters across GL, Purchasing, and Payables modules.
    
    OPTIMIZED MULTI-MODULE APPROACH:
      - Supports GL (ledgers, periods), Purchasing (vendors, PO numbers), Payables (suppliers, invoices)
      - Analyzes WHERE clause to distinguish user inputs from hardcoded business logic
      - Module-aware prompting with relevant examples
      - Smart fallback for each module
      - No hardcoded parameter whitelists - works with any user parameter

    Flow:
      1) Detect module from tables (GL/Purchasing/Payables)
      2) Extract WHERE clause and user parameter hints
      3) Build module-aware prompt with focused examples
      4) Generate validation queries via LLM
      5) Parse and validate results
      6) Smart module-aware fallback if needed
    """
    import re
    import json
    import logging
    import inspect
    from datetime import datetime
    from pathlib import Path
    from llm_utils import GenConfig, generate

    logger = logging.getLogger(__name__)
    
    def log_with_context(msg: str, level: str = "info"):
        """Log with timestamp and line number"""
        frame = inspect.currentframe().f_back
        line_no = frame.f_lineno
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_msg = f"[{timestamp}] [Line {line_no}] {msg}"
        getattr(logger, level)(log_msg)

    # --------------------- helpers ---------------------
    # REMOVED: ALLOWED_PARAMS whitelist - now module-aware and dynamic
    MONTH_NAMES = {
        "january": "JAN", "jan": "JAN",
        "february": "FEB", "feb": "FEB",
        "march": "MAR", "mar": "MAR",
        "april": "APR", "apr": "APR",
        "may": "MAY",  "may": "MAY",
        "june": "JUN",  "jun": "JUN",
        "july": "JUL",  "jul": "JUL",
        "august": "AUG","aug": "AUG",
        "september": "SEP","sep": "SEP",
        "october": "OCT","oct": "OCT",
        "november": "NOV","nov": "NOV",
        "december": "DEC","dec": "DEC",
    }

    def _param_canon(p: str | None) -> str | None:
        """
        Normalize parameter names to canonical forms.
        Now supports GL, Purchasing, and Payables parameters.
        """
        if not p:
            return None
        s = p.strip().lower()
        
        # GL parameters
        if "ledger" in s:
            return "ledger_name"
        if "period" in s:
            return "period"
        
        # Purchasing parameters
        if "vendor" in s and "name" in s:
            return "vendor_name"
        if "po" in s and ("number" in s or "num" in s):
            return "po_number"
        if "buyer" in s:
            return "buyer_name"
        
        # Payables parameters
        if "supplier" in s:
            return "supplier_name"
        if "invoice" in s and ("number" in s or "num" in s):
            return "invoice_number"
        
        # Common parameters
        if "operating" in s and "unit" in s:
            return "operating_unit"
        
        # Return as-is if no match (don't filter out)
        return p.strip()

    def _validate_parameters_in_question(items: list[ValidationQueryItem], question: str) -> list[ValidationQueryItem]:
        """
        Hallucination Detection: Ensure extracted parameters actually appear in the question.
        Prevents LLM from inventing parameters not mentioned by user.
        """
        validated = []
        question_lower = question.lower()
        
        for item in items:
            orig_param = (item.original_parameters or "").strip('"\'')
            if not orig_param:
                # No original parameter specified - keep it (might be inferred from SQL)
                validated.append(item)
                continue
            
            # Check if parameter appears in question (fuzzy match)
            # Split parameter into words and check if any significant word appears
            param_words = orig_param.lower().split()
            significant_words = [word for word in param_words if len(word) > 2]  # Skip short words like "of", "in"
            
            if any(word in question_lower for word in significant_words):
                validated.append(item)
            else:
                logger.warning(f"🚫 Dropping hallucinated parameter: '{orig_param}' (not found in question)")
        
        return validated

    def _validate_sql_safety(items: list[ValidationQueryItem], cards: dict) -> list[ValidationQueryItem]:
        """
        SQL Safety Validation: Validate SQL syntax, schema references, and prevent injection.
        Ensures only safe, valid validation queries are returned.
        """
        validated = []
        
        for item in items:
            sql = (item.sql or "").strip()
            
            if not sql:
                logger.warning(f"🚫 Dropping item with empty SQL")
                continue
            
            # Check 1: Must be SELECT only
            sql_upper = sql.upper()
            if not sql_upper.startswith("SELECT"):
                logger.warning(f"🚫 Dropping non-SELECT query: {sql[:50]}...")
                continue
            
            # Check 2: No dangerous keywords (SQL injection prevention)
            dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE", "EXEC", "EXECUTE"]
            if any(kw in sql_upper for kw in dangerous):
                logger.warning(f"🚫 Dropping dangerous query with keyword: {sql[:50]}...")
                continue
            
            # Check 3: Table exists in schema
            # Extract table names from SQL (FROM and JOIN clauses)
            tables_in_sql = re.findall(r'\bFROM\s+([A-Z_]+)|\bJOIN\s+([A-Z_]+)', sql_upper, re.IGNORECASE)
            table_names = [t[0] or t[1] for t in tables_in_sql]
            
            if table_names:
                # Check if at least one table exists in schema
                unknown_tables = [t for t in table_names if t.upper() not in cards]
                if unknown_tables and len(unknown_tables) == len(table_names):
                    # All tables are unknown - drop it
                    logger.warning(f"🚫 Dropping query with unknown tables: {unknown_tables}")
                    continue
                elif unknown_tables:
                    # Some tables unknown - log warning but keep it
                    logger.warning(f"⚠️ Query references unknown tables: {unknown_tables} (keeping anyway)")
            
            # Check 4: Must contain WHERE clause (validation queries should filter)
            if "WHERE" not in sql_upper:
                logger.warning(f"⚠️ Validation query missing WHERE clause: {sql[:50]}... (keeping anyway)")
            
            # All checks passed
            validated.append(item)
        
        return validated

    def _filter_allowed(items: list[ValidationQueryItem], module: str) -> list[ValidationQueryItem]:
        """
        Filter validation queries based on detected module.
        Now module-aware - no hardcoded whitelist.
        """
        out = []
        for it in items:
            # If parameter is None (from Pass C fallback), try to infer from SQL
            if it.parameter is None and it.sql:
                # Try to infer parameter name from table name in SQL
                sql_upper = it.sql.upper()
                if 'GL_LEDGERS' in sql_upper:
                    it.parameter = 'ledger_name'
                elif 'GL_PERIODS' in sql_upper:
                    it.parameter = 'period'
                elif 'PO_VENDORS' in sql_upper:
                    it.parameter = 'vendor_name'
                elif 'AP_SUPPLIERS' in sql_upper:
                    it.parameter = 'supplier_name'
                elif 'PO_HEADERS_ALL' in sql_upper:
                    it.parameter = 'po_number'
                elif 'AP_INVOICES_ALL' in sql_upper:
                    it.parameter = 'invoice_number'
                elif 'HR_OPERATING_UNITS' in sql_upper:
                    it.parameter = 'operating_unit'
                elif 'HR_EMPLOYEES' in sql_upper:
                    it.parameter = 'buyer_name'
            
            # Normalize parameter name
            canon = _param_canon(it.parameter)
            if canon and (it.sql or "").strip():
                it.parameter = canon
                out.append(it)
        return out

    def _extract_ledger_token_from_sql(sql: str) -> tuple[str | None, str | None]:
        """
        Return (like_token, eq_phrase)
          like_token: from UPPER(LED.NAME) LIKE UPPER('%Vision%Operations%USA%')
          eq_phrase:  from LED.NAME = 'Vision Operations (USA)'
        """
        if not sql:
            return (None, None)
        m_like = re.search(
            r"UPPER\(\s*(?:LED(?:GERS)?|GL_LEDGERS)\s*\.\s*NAME\s*\)\s*LIKE\s*UPPER\(\s*'%([^']+)%'\s*\)",
            sql, flags=re.IGNORECASE
        )
        like_token = m_like.group(1) if m_like else None

        m_eq = re.search(
            r"(?:LED(?:GERS)?|GL_LEDGERS)\s*\.\s*NAME\s*=\s*'([^']+)'",
            sql, flags=re.IGNORECASE
        )
        eq_phrase = m_eq.group(1) if m_eq else None
        return (like_token, eq_phrase)

    def _phrase_to_like_token(phrase: str) -> str:
        """
        'Vision Operations (USA)' -> 'Vision%Operations%USA'
        """
        cleaned = re.sub(r"[^A-Za-z0-9]+", " ", phrase or "").strip()
        if not cleaned:
            return ""
        return "%".join(w for w in cleaned.split() if w)

    def _extract_periods_from_question(q: str) -> list[tuple[str, str]]:
        """
        Return list of (MON, YY) pairs robustly from variants in the question:
          'Jan 03', 'January 2003', 'Jan-2003', 'Jun 2003', etc.
        """
        if not q:
            return []
        text = q.lower()

        out: list[tuple[str, str]] = []

        # Patterns:
        #  A) Month Word + space/hyphen + 2/4-digit year
        for m in re.finditer(r"\b([a-z]{3,9})\s*[- ]\s*(\d{2}|\d{4})\b", text, flags=re.IGNORECASE):
            mon_word = m.group(1).lower()
            yr = m.group(2)
            if mon_word in MONTH_NAMES:
                MON = MONTH_NAMES[mon_word]
                YY = yr[-2:]
                out.append((MON, YY))

        #  B) Abbrev Month (no space) + 2-digit year (e.g., 'JAN03')
        for m in re.finditer(r"\b([a-z]{3})(\d{2})\b", text, flags=re.IGNORECASE):
            mon_abbrev = m.group(1).lower()
            yy = m.group(2)
            if mon_abbrev in MONTH_NAMES:
                MON = MONTH_NAMES[mon_abbrev]
                out.append((MON, yy))

        # Deduplicate while preserving order
        seen = set()
        uniq: list[tuple[str, str]] = []
        for t in out:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    def _display_from_like_token(token: str) -> str:
        # "Vision%Operations%USA" -> "Vision Operations USA"
        disp = re.sub(r'%+', ' ', token or "").strip()
        return disp or (token or "")

    def _strip_cte_names(sql: str, tables: list[str]) -> list[str]:
        ctes = set(re.findall(r'\bwith\s+([a-zA-Z0-9_#\$]+)\s+as\s*\(', sql or "", flags=re.IGNORECASE))
        return [t for t in (tables or []) if t.upper() not in {c.upper() for c in ctes}]

    # Parser that handles (1) fenced ```sql blocks and (2) unfenced outputs starting with SELECT
    def _parse_validation_queries_robust(raw: str) -> list[ValidationQueryItem]:
        if not raw:
            return []
        lines = [l.rstrip() for l in raw.splitlines()]
        items: list[ValidationQueryItem] = []

        # Pass A: fenced
        cur = {"original_parameters": None, "parameter": None, "description": None, "purpose": None, "sql": ""}
        i = 0

        def flush():
            if cur.get("sql"):
                # Strip semicolons and trailing backticks (markdown artifacts)
                cleaned_sql = cur.get("sql").strip().strip(';').rstrip('`').strip()
                items.append(ValidationQueryItem(
                    original_parameters=cur.get("original_parameters"),
                    parameter=cur.get("parameter"),
                    description=cur.get("description"),
                    purpose=cur.get("purpose"),
                    sql=cleaned_sql
                ))

        fenced_found = False
        while i < len(lines):
            L = lines[i]
            low = L.lower().strip()
            if low.startswith("-- original parameters:"):
                if cur.get("sql"):
                    flush()
                    cur = {"original_parameters": None, "parameter": None, "description": None, "purpose": None, "sql": ""}
                cur["original_parameters"] = L.split(":", 1)[1].strip() if ":" in L else None
            elif low.startswith("-- parameter:"):
                cur["parameter"] = L.split(":", 1)[1].strip() if ":" in L else None
            elif low.startswith("-- description:"):
                cur["description"] = L.split(":", 1)[1].strip() if ":" in L else None
            elif low.startswith("-- purpose:"):
                cur["purpose"] = L.split(":", 1)[1].strip() if ":" in L else None
            elif low.startswith("```sql"):
                fenced_found = True
                i += 1
                buf = []
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    # Check if line contains inline backticks (e.g., "...SQL```")
                    line = lines[i]
                    if "```" in line:
                        # Split at backticks and take only the part before them
                        line = line.split("```")[0]
                    buf.append(line)
                    i += 1
                cur["sql"] = "\n".join(buf).strip()
            i += 1
        if fenced_found:
            if cur.get("sql"):
                flush()
            return items

        # Pass B: unfenced header + SELECT
        items = []
        cur = {"original_parameters": None, "parameter": None, "description": None, "purpose": None, "sql": ""}
        i = 0

        def is_header(s: str) -> bool:
            s = s.strip().lower()
            return s.startswith("-- parameter:") or s.startswith("-- original parameters:") or s.startswith("-- description:") or s.startswith("-- purpose:")

        while i < len(lines):
            L = lines[i]
            low = L.strip().lower()

            if low.startswith("-- original parameters:"):
                if cur.get("sql"):
                    # Strip semicolons and trailing backticks (markdown artifacts)
                    cleaned_sql = cur.get("sql").strip().strip(';').rstrip('`').strip()
                    items.append(ValidationQueryItem(
                        original_parameters=cur.get("original_parameters"),
                        parameter=cur.get("parameter"),
                        description=cur.get("description"),
                        purpose=cur.get("purpose"),
                        sql=cleaned_sql
                    ))
                    cur = {"original_parameters": None, "parameter": None, "description": None, "purpose": None, "sql": ""}
                cur["original_parameters"] = L.split(":", 1)[1].strip() if ":" in L else None

            elif low.startswith("-- parameter:"):
                if cur.get("sql"):
                    # Strip semicolons and trailing backticks (markdown artifacts)
                    cleaned_sql = cur.get("sql").strip().strip(';').rstrip('`').strip()
                    items.append(ValidationQueryItem(
                        original_parameters=cur.get("original_parameters"),
                        parameter=cur.get("parameter"),
                        description=cur.get("description"),
                        purpose=cur.get("purpose"),
                        sql=cleaned_sql
                    ))
                    cur = {"original_parameters": None, "parameter": None, "description": None, "purpose": None, "sql": ""}
                cur["parameter"] = L.split(":", 1)[1].strip() if ":" in L else None

            elif low.startswith("-- description:"):
                cur["description"] = L.split(":", 1)[1].strip() if ":" in L else None

            elif low.startswith("-- purpose:"):
                cur["purpose"] = L.split(":", 1)[1].strip() if ":" in L else None

            elif re.match(r'^\s*select\b', L, flags=re.IGNORECASE):
                buf = [L]
                j = i + 1
                while j < len(lines):
                    if not lines[j].strip():
                        break
                    if is_header(lines[j]):
                        break
                    buf.append(lines[j])
                    j += 1
                cur["sql"] = "\n".join(buf).strip()
                i = j - 1
            i += 1

        if cur.get("sql"):
            # Strip semicolons and trailing backticks (markdown artifacts)
            cleaned_sql = cur.get("sql").strip().strip(';').rstrip('`').strip()
            items.append(ValidationQueryItem(
                original_parameters=cur.get("original_parameters"),
                parameter=cur.get("parameter"),
                description=cur.get("description"),
                purpose=cur.get("purpose"),
                sql=cleaned_sql
            ))
        
        if items:
            return items
        
        # Pass C: Raw SQL without headers (fallback for when LLM skips comment headers)
        # Extract all SELECT statements and create minimal ValidationQueryItem objects
        items = []
        i = 0
        while i < len(lines):
            L = lines[i]
            # Skip template placeholders like [column_name], [master_table], [pattern]
            if '[column_name]' in L or '[master_table]' in L or '[pattern]' in L:
                i += 1
                continue
            
            if re.match(r'^\s*select\b', L, flags=re.IGNORECASE):
                buf = [L]
                j = i + 1
                # Collect multi-line SELECT statement
                while j < len(lines):
                    next_line = lines[j].strip()
                    # Stop at empty line, next SELECT, or comment header
                    if not next_line:
                        break
                    if re.match(r'^\s*select\b', next_line, flags=re.IGNORECASE):
                        break
                    if next_line.startswith('--'):
                        break
                    buf.append(lines[j])
                    j += 1
                
                sql = "\n".join(buf).strip().strip(';').rstrip('`').strip()
                
                # Only add if it's a valid SELECT statement (not a template)
                if sql and 'SELECT' in sql.upper() and '[' not in sql:
                    items.append(ValidationQueryItem(
                        original_parameters=None,
                        parameter=None,
                        description=None,
                        purpose=None,
                        sql=sql
                    ))
                i = j
            else:
                i += 1
        
        return items

    # --------------------- model + context extraction ---------------------
    tok, mdl = _get_model(CFG)

    # Extract tables and identify module
    used_tables = _tables_from_sql(request.generated_sql)
    used_tables = _strip_cte_names(request.generated_sql, used_tables)
    module = _identify_module_from_tables(used_tables)
    
    log_with_context(f"🔍 Detected module(s): {module}")
    log_with_context(f"📊 Tables used: {', '.join(used_tables)}")
    
    # Extract WHERE clause for focused analysis
    where_clause = _extract_where_clause(request.generated_sql or "")
    log_with_context(f"🎯 WHERE clause extracted: {len(where_clause)} characters")
    
    # Extract user parameter hints from question
    user_hints = _extract_user_parameters_from_question(request.original_question or "")
    log_with_context(f"💡 User parameter hints: {user_hints}")
    
    # Legacy GL-specific seeds (still used for fallback)
    like_token, eq_phrase = _extract_ledger_token_from_sql(request.generated_sql or "")
    ledger_like = like_token or (_phrase_to_like_token(eq_phrase) if eq_phrase else None)
    period_hits = _extract_periods_from_question(request.original_question or "")

    # --------------------- schema grounding (multi-module master tables) ---------------------
    try:
        # Define master tables for each module (for validation queries)
        master_tables_by_module = {
            "GL": {"GL_LEDGERS", "GL_PERIODS", "GL_CODE_COMBINATIONS"},
            "Purchasing": {"PO_VENDORS", "PO_HEADERS_ALL", "HR_OPERATING_UNITS", "HR_EMPLOYEES"},
            "Payables": {"AP_SUPPLIERS", "AP_INVOICES_ALL", "HR_OPERATING_UNITS", "AP_TERMS"}
        }
        
        # Collect relevant master tables based on detected module
        relevant_master_tables = set()
        for mod in module.split(", "):
            if mod in master_tables_by_module:
                relevant_master_tables.update(master_tables_by_module[mod])
        
        # Filter to tables actually used in the query
        base_tables = [t for t in used_tables if t.upper() in relevant_master_tables]
        
        log_with_context(f"📚 Master tables for validation: {', '.join(base_tables) if base_tables else 'None'}")
        
        schema_block = _schema_block_for_tables(base_tables, Path(CFG.tables_dir), max_cols_per_table=8) if base_tables else ""
    except Exception as e:
        log_with_context(f"⚠️ Schema extraction failed: {e}", level="warning")
        schema_block = ""

    # --------------------- optimized multi-module prompt ---------------------
    # Build WHERE clause highlight (simplified)
    where_highlight = f"\nWHERE: {where_clause}\n" if where_clause else ""
    
    # Build table context (simplified)
    table_context = f"Module: {module}\n" if module else ""

    # Enhanced comprehensive examples based on module (2-3 examples per module)
    # Simplified format: Just Question + SQL (following XiYan template)
    if "Payables" in module:
        examples_block = """--- Example 1 ---
Question: "Show invoices for supplier ABC Corporation"
SELECT DISTINCT VENDOR_NAME FROM AP_SUPPLIERS WHERE LOWER(VENDOR_NAME) LIKE LOWER('%abc%corporation%')

--- Example 2 ---
Question: "Show payment details for invoice INV-2024-001"
SELECT DISTINCT INVOICE_NUM FROM AP_INVOICES_ALL WHERE LOWER(INVOICE_NUM) LIKE LOWER('%inv%2024%001%')


--- Example 3 ---
Question: "Show invoices for vision Operations operating unit"
SELECT DISTINCT NAME FROM HR_OPERATING_UNITS WHERE LOWER(NAME) LIKE LOWER('%vision%operations%')
""".strip()
    elif "Purchasing" in module:
        examples_block = """--- Example 1 ---
Question: "Show purchase orders for vendor XYZ Supplies Inc"
SELECT DISTINCT VENDOR_NAME FROM PO_VENDORS WHERE LOWER(VENDOR_NAME) LIKE LOWER('%xyz%supplies%inc%')

--- Example 2 ---
Question: "Show details for purchase order PO-12345"
SELECT DISTINCT SEGMENT1 FROM PO_HEADERS_ALL WHERE LOWER(SEGMENT1) LIKE LOWER('%po%12345%')


--- Example 3 ---
Question: "Show POs created by buyer John Smith"
SELECT DISTINCT FULL_NAME FROM HR_EMPLOYEES WHERE LOWER(FULL_NAME) LIKE LOWER('%john%smith%')
""".strip()
    else:  # GL
        examples_block = """--- Example 1 ---
Question: "Show balances for Vision Operations ledger"
SELECT DISTINCT NAME FROM GL_LEDGERS WHERE LOWER(NAME) LIKE LOWER('%vision%operations%')

--- Example 2 ---
Question: "Show balances for period Jan-03"
SELECT DISTINCT PERIOD_NAME FROM GL_PERIODS WHERE LOWER(PERIOD_NAME) LIKE LOWER('%jan%03%')


--- Example 3 ---
Question: "Show balances for operating unit vision Operations"
SELECT DISTINCT NAME FROM HR_OPERATING_UNITS WHERE LOWER(NAME) LIKE LOWER('%vision%operations%')
""".strip()

    # Build reference block (rules + examples)
    rules_block = """RULES:
✅ Extract: Parameters from user question (names, numbers, dates)
❌ Ignore: Hardcoded filters (STATUS='APPROVED', TYPE='STANDARD')
✅ Use: Master tables only (GL_LEDGERS, AP_SUPPLIERS, PO_VENDORS, etc.)
❌ Skip: Transaction tables (GL_BALANCES, AP_INVOICE_LINES, etc.)

OUTPUT FORMAT (include ALL 4 comment lines):
-- Original Parameters: [phrase from question]
-- Parameter: [name]
-- Description: [what it is]
-- Purpose: [why validate]
```sql
SELECT DISTINCT [column] FROM [table] WHERE LOWER([column]) LIKE LOWER('%pattern%')
```

MASTER TABLES:
GL: GL_LEDGERS.NAME, GL_PERIODS.PERIOD_NAME
Purchasing: PO_VENDORS.VENDOR_NAME, PO_HEADERS_ALL.SEGMENT1, HR_EMPLOYEES.FULL_NAME
Payables: AP_SUPPLIERS.VENDOR_NAME, AP_INVOICES_ALL.INVOICE_NUM""".strip()
    
    reference_block = f"{rules_block}\n\n{examples_block}".strip()
    
    # Follow XiYan template structure (same as /inference endpoint)
    validation_prompt = f"""You are a PostgreSQL expert. Extract validation queries for user parameters.
Only output validation SQL queries. Use master tables only.

[User Question]
{request.original_question}

[Database Schema]
{schema_block}

[Reference Information]
{reference_block}

[User Question]
{request.original_question}

```sql
""".strip()

    # --------------------- generation ---------------------
    # Optimized for extraction task (not SQL generation)
    gen_cfg = GenConfig(
        model_id=CFG.model_id,
        max_new_tokens=int(request.max_new_tokens or 800),  # Validation queries are shorter than SQL
        temperature=float(request.temperature or 0.0),  # Deterministic extraction (not creative generation)
        top_p=float(request.top_p or 1.0),  # No nucleus sampling needed for extraction
        repetition_penalty=1.0,
    )
    
    # LOG: Full prompt being sent to LLM
    log_with_context("=" * 80)
    log_with_context("📤 FULL PROMPT SENT TO LLM (Initial):")
    log_with_context("=" * 80)
    log_with_context(f"Prompt length: {len(validation_prompt)} characters")
    log_with_context("vvvv FULL PROMPT START vvvv")
    log_with_context(validation_prompt)
    log_with_context("^^^^ FULL PROMPT END ^^^^")
    log_with_context("=" * 80)
    
    validation_response = (generate(mdl, tok, validation_prompt, gen_cfg) or "").strip()
    
    # LOG 1: Model's raw response
    log_with_context("=" * 80)
    log_with_context("📝 LLM RAW RESPONSE - VALIDATION QUERIES:")
    log_with_context("=" * 80)
    if validation_response:
        log_with_context(f"Response length: {len(validation_response)} characters")
        log_with_context("vvvv FULL LLM RESPONSE START vvvv")
        log_with_context(validation_response)
        log_with_context("^^^^ FULL LLM RESPONSE END ^^^^")
    else:
        log_with_context("⚠️ WARNING: LLM returned EMPTY response!", level="warning")
    log_with_context("=" * 80)

    # Ultra-short greedy retry if needed
    if not validation_response:
        # Build module-specific retry prompt (following XiYan template)
        if "Payables" in module:
            retry_example = """--- Example ---
Question: "Show payment details for invoice INV-001"
```sql
-- Original Parameters: "INV-001"
-- Parameter: invoice_number
-- Description: Invoice number from user
-- Purpose: Confirm invoice exists
SELECT DISTINCT INVOICE_NUM FROM AP_INVOICES_ALL WHERE LOWER(INVOICE_NUM) LIKE LOWER('%inv%001%')
```""".strip()
        elif "Purchasing" in module:
            retry_example = """--- Example ---
Question: "Show details for purchase order PO-12345"
```sql
-- Original Parameters: "PO-12345"
-- Parameter: po_number
-- Description: PO number from user
-- Purpose: Confirm PO exists
SELECT DISTINCT SEGMENT1 FROM PO_HEADERS_ALL WHERE LOWER(SEGMENT1) LIKE LOWER('%po%12345%')
```""".strip()
        else:  # GL
            retry_example = """--- Example ---
Question: "Show balances for Vision Operations ledger"
```sql
-- Original Parameters: "Vision Operations"
-- Parameter: ledger_name
-- Description: Ledger name from user
-- Purpose: Confirm ledger exists
SELECT DISTINCT NAME FROM GL_LEDGERS WHERE LOWER(NAME) LIKE LOWER('%vision%operations%')
```""".strip()
        
        # Follow XiYan template for retry (simplified version)
        short_prompt = f"""You are a PostgreSQL expert. Extract validation queries for user parameters.

[User Question]
{request.original_question}

[Reference Information]
{retry_example}

[User Question]
{request.original_question}

```sql""".strip()
        
        # LOG: Retry prompt
        log_with_context("=" * 80)
        log_with_context("🔄 RETRY PROMPT (Initial was empty):")
        log_with_context("=" * 80)
        log_with_context(f"Retry prompt length: {len(short_prompt)} characters")
        log_with_context("vvvv RETRY PROMPT START vvvv")
        log_with_context(short_prompt)
        log_with_context("^^^^ RETRY PROMPT END ^^^^")
        log_with_context("=" * 80)
        
        gen_cfg.temperature = 0.0
        gen_cfg.top_p = 1.0
        gen_cfg.max_new_tokens = min(gen_cfg.max_new_tokens, 512)
        validation_response = (generate(mdl, tok, short_prompt, gen_cfg) or "").strip()
    
        if validation_response:
            log_with_context("=" * 80)
            log_with_context("🔄 RETRY LLM RESPONSE:")
            log_with_context("=" * 80)
            log_with_context(f"Retry response length: {len(validation_response)} characters")
            log_with_context("vvvv RETRY RESPONSE START vvvv")
            log_with_context(validation_response)
            log_with_context("^^^^ RETRY RESPONSE END ^^^^")
            log_with_context("=" * 80)
        else:
            log_with_context("⚠️ WARNING: RETRY also returned EMPTY response!", level="warning")


    # --------------------- parse + normalize + filter ---------------------
    log_with_context("=" * 80)
    log_with_context("🔍 PARSING LLM RESPONSE:")
    log_with_context("=" * 80)
    items = _parse_validation_queries_robust(validation_response)
    log_with_context(f"✅ Parsed {len(items)} items from LLM response")
    
    if items:
        log_with_context("📋 LLM INFERRED VALIDATION QUERIES (before filtering):")
        for idx, item in enumerate(items, 1):
            log_with_context(f"  Item #{idx}:")
            log_with_context(f"    - parameter: {item.parameter}")
            log_with_context(f"    - original_parameters: {item.original_parameters}")
            log_with_context(f"    - description: {item.description}")
            log_with_context(f"    - purpose: {item.purpose}")
            sql_preview = (item.sql or "")[:200].replace("\n", " ")
            log_with_context(f"    - sql: {sql_preview}{'...' if len(item.sql or '') > 200 else ''}")
    else:
        log_with_context("⚠️ WARNING: Parser returned 0 items from LLM response!", level="warning")
    log_with_context("=" * 80)

    log_with_context(f"🔍 Filtering items (module-aware for {module})...")
    items_before_filter = len(items)
    items = _filter_allowed(items, module)
    log_with_context(f"✅ After filtering: {len(items)} items (removed {items_before_filter - len(items)})")

    # --------------------- hallucination detection ---------------------
    log_with_context(f"🔍 Validating parameters appear in question...")
    items_before_hallucination = len(items)
    items = _validate_parameters_in_question(items, request.original_question)
    log_with_context(f"✅ After hallucination check: {len(items)} items (removed {items_before_hallucination - len(items)} hallucinated)")

    # --------------------- sql safety validation ---------------------
    log_with_context(f"🔍 Validating SQL safety (syntax, schema, injection prevention)...")
    items_before_safety = len(items)
    cards = load_cards(Path(CFG.tables_dir))
    items = _validate_sql_safety(items, cards)
    log_with_context(f"✅ After safety validation: {len(items)} items (removed {items_before_safety - len(items)} unsafe)")

    # --------------------- smart multi-module fallback ---------------------
    log_with_context("=" * 80)
    log_with_context("🔧 SMART FALLBACK CHECK (Module-Aware):")
    log_with_context("=" * 80)
    
    # Check if LLM provided any results
    if not items:
        log_with_context("⚠️ LLM returned no validation queries, using smart fallback", level="warning")
        fallback_items = _generate_smart_fallback_queries(
            question=request.original_question,
            sql=request.generated_sql,
            tables=used_tables,
            module=module,
            hints=user_hints,
            ledger_like=ledger_like,
            period_hits=period_hits
        )
        items.extend(fallback_items)
        log_with_context(f"⚠️ Added {len(fallback_items)} fallback validation queries", level="warning")
    else:
        log_with_context(f"✅ LLM provided {len(items)} validation queries - no fallback needed")
    
    log_with_context("=" * 80)

    # --------------------- final ---------------------
    if not items:
        response = ValidationQueryResponse(
            validation_queries=[],
            success=True,
            message=f"No user parameters found requiring validation for {module} module(s)"
        )
    else:
        response = ValidationQueryResponse(
            validation_queries=items,
            success=True,
            message=f"Extracted {len(items)} validation queries for {module} module(s)"
        )
    
    # LOG 2: Final JSON response being sent back
    log_with_context("=" * 80)
    log_with_context("📤 FINAL JSON RESPONSE - VALIDATION QUERIES:")
    log_with_context("=" * 80)
    response_dict = response.dict()
    log_with_context(f"Module(s): {module}")
    log_with_context(f"Total validation queries in response: {len(response_dict.get('validation_queries', []))}")
    
    # Log each validation query
    for idx, vq in enumerate(items, 1):
        log_with_context(f"  Query {idx}: {vq.parameter} - {vq.original_parameters}")
    
    log_with_context("vvvv FINAL JSON START vvvv")
    log_with_context(json.dumps(response_dict, indent=2, ensure_ascii=False))
    log_with_context("^^^^ FINAL JSON END ^^^^")
    log_with_context("=" * 80)
    
    return response



def run():
    import logging
    logging.basicConfig(level=logging.INFO)
    print("Starting FastAPI ARAG 32B server (finetuned model)...")
    uvicorn.run("fast_api_arag_32b:app", host="0.0.0.0", port=8002)

if __name__ == "__main__":
    run()
