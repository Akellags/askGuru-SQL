import json
from typing import List, Dict, Any, Optional
from .config import settings

def load_mschema(file_path: str) -> List[Dict[str, Any]]:
    """Load M-Schema from a line-delimited JSON file."""
    mschema = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    mschema.append(json.loads(line))
    except Exception as e:
        print(f"Error loading M-Schema: {e}")
    return mschema

def format_mschema_text(tables_data: List[Dict[str, Any]]) -> str:
    """Format M-Schema dictionary into the text format expected by the model."""
    lines = []
    for table in tables_data:
        # Check if table has essential columns
        essential_cols = [c for c in table['columns'] if c.get('is_primary') or c.get('is_essential')]
        
        lines.append(f"{table['table_name']} (")
        for col in table['columns']:
            prefix = "**[ESSENTIAL]** " if (col.get('is_primary') or col.get('is_essential')) else ""
            desc = f" -- {col.get('description', '')}" if col.get('description') else ""
            lines.append(f"  {prefix}{col['name']} {col['type']}{desc}")
        lines.append(")")
        lines.append("") # Spacer
    
    return "\n".join(lines)

def get_join_hints(mschema: List[Dict[str, Any]], requested_tables: List[str]) -> str:
    """Extract join hints for requested tables."""
    hints = []
    table_names = [t.lower() for t in requested_tables]
    for table in mschema:
        if table['table_name'].lower() in table_names:
            for fk in table.get('foreign_keys', []):
                if fk['ref_table'].lower() in table_names:
                    hints.append(f"  - {table['table_name']}.{fk['column']} = {fk['ref_table']}.{fk['ref_column']}")
    
    if not hints:
        return ""
    return "\nJOIN HINTS:\n" + "\n".join(list(set(hints)))

def build_llama_prompt(request: Any, mschema: List[Dict[str, Any]], schema_text: str, rag_context: Optional[Dict[str, Any]] = None) -> str:
    """Build the prompt for LLaMA-3.1 model."""
    enriched_question = build_enriched_question(
        request.question, 
        request.filters, 
        request.group_columns, 
        request.columns_list
    )
    
    # Use RAG evidence if available, otherwise fallback to static join hints
    if rag_context and rag_context.get("evidence"):
        reference_info = f"[Evidence]\n{rag_context['evidence']}\n"
    else:
        join_hints = get_join_hints(mschema, request.tables or [])
        reference_info = f"{join_hints}\n"

    # Format few-shots from RAG
    fewshot_text = ""
    if rag_context and rag_context.get("fewshots"):
        fewshot_text = "\n[Few-Shot Examples]\n"
        for fs in rag_context["fewshots"]:
            fewshot_text += f"Question: {fs.get('intent')}\n"
            fewshot_text += f"SQL: {fs.get('sql')}\n\n"

    return f"""You are a Oracle EBS expert. Generate executable SQL based on the user's question.
Only output SQL query. Do not invent columns - use only those in the schema.
CRITICAL: Columns marked with **[ESSENTIAL]** are mandatory for proper joins and aggregations - prioritize them.

[User Question]
{enriched_question}

[Database Schema]
{schema_text}
{reference_info}
{fewshot_text}

[Reference Information]
[Rules]
- **CRITICAL**: Examples marked with [* VERY SIMILAR] are semantically matched to your question. Follow their SQL structure, JOIN patterns, WHERE logic, aggregation strategy, and CTE usage EXACTLY. Do NOT simplify or refactor them - they are proven patterns that work for this data model.
- Examples marked with * are most similar to the current question - follow their patterns and structure closely. Mimic their JOIN order, subquery approach (CTE vs NOT EXISTS vs LEFT JOIN), aggregation, and filtering logic.
- **MANDATORY**: Always include all JOIN key columns in the SELECT clause. Every ID or key used to JOIN tables must be present in the final result set.
- Do not invent columns; use only those in [Database Schema].
- When computing closing balance, use this expression exactly:
  NVL(BEGIN_BALANCE_DR,0) - NVL(BEGIN_BALANCE_CR,0)
  + NVL(PERIOD_NET_DR,0) - NVL(PERIOD_NET_CR,0)
- Do NOT reference a column alias inside CASE unless it is computed in a subquery.
  EITHER inline the full expression inside CASE (preferred),
  OR compute it in a subquery and reference it in the outer SELECT.
- Always guard divisions:  / NULLIF(ABS(<denominator>), 0)
- Prefer PERIOD_NAME-based filters (e.g., 'Jan-03','Jun-03'), not literal START_DATEs.
- CRITICAL: Month abbreviations in PERIOD_NAME must ALWAYS be in title case format: 'Jan-03', 'Feb-03', 'Mar-03', 'Apr-03', 'May-03', 'Jun-03', 'Jul-03', 'Aug-03', 'Sep-03', 'Oct-03', 'Nov-03', 'Dec-03'. Do NOT use uppercase (MAR-03, APR-03) or lowercase (mar-03, apr-03).
- When in any join that has MTL_SYSTEM_ITEMS table must necessary have 2 joins.

Follow JOIN HINTS. Do not invent columns. Guard divisions with NULLIF.

Return **Oracle SQL only**. No markdown. No explanation.
```sql"""

def get_filtered_schema(mschema: List[Dict[str, Any]], requested_tables: List[str]) -> str:
    """Filter the M-Schema for requested tables and format as text."""
    if not requested_tables:
        return "No tables requested for static filtering."
    
    table_names = [t.upper().strip() for t in requested_tables]
    filtered_data = [t for t in mschema if t['table_name'].upper().strip() in table_names]
    
    if not filtered_data:
        return f"Warning: None of the requested tables {requested_tables} found in schema."
    
    return format_mschema_text(filtered_data)

def get_filtered_schema_from_rag(rag_context: Dict[str, Any], mschema: List[Dict[str, Any]]) -> str:
    """Format M-Schema based on RAG selected tables and columns, using MSCHEMA for full details."""
    selected_tables = [t.upper() for t in rag_context.get("tables", [])]
    if not selected_tables:
        return "No tables identified by RAG for this question."
        
    selected_cols_map = rag_context.get("columns", {})
    
    # Filter mschema to only include RAG-selected tables
    filtered_mschema = []
    for table_card in mschema:
        tname = table_card['table_name'].upper()
        if tname in selected_tables:
            # Optionally further filter columns based on RAG selected columns
            # but usually it's better to provide all columns from the selected tables
            # or at least the ones RAG thinks are important + ESSENTIAL ones.
            filtered_mschema.append(table_card)
            
    if not filtered_mschema:
        return f"Warning: RAG selected tables {selected_tables} but they were not found in MSCHEMA."
        
    return format_mschema_text(filtered_mschema)

def build_enriched_question(question: str, filters: Optional[List[str]] = None, group_columns: Optional[List[str]] = None, columns_list: Optional[List[str]] = None) -> str:
    """Helper to build a detailed question string."""
    parts = [question]
    if filters:
        f_str = filters if isinstance(filters, str) else ", ".join(filters)
        parts.append(f"Filters: {f_str}")
    if group_columns:
        g_str = group_columns if isinstance(group_columns, str) else ", ".join(group_columns)
        parts.append(f"Group by: {g_str}")
    if columns_list:
        c_str = columns_list if isinstance(columns_list, str) else ", ".join(columns_list)
        parts.append(f"Required columns: {c_str}")
    return " | ".join(parts)

def build_sqlcoder_prompt(request: Any, mschema: List[Dict[str, Any]], schema_text: str) -> str:
    """Build the prompt for SQLCoder-70B model."""
    enriched_question = build_enriched_question(
        request.question, 
        request.filters, 
        None,
        request.columns_list
    )
    
    join_hints = get_join_hints(mschema, request.tables or [])
    
    return f"""### Task
Generate a SQL query to answer the following question: `{enriched_question}`

### Database Schema
{schema_text}
{join_hints}

### SQL Query
"""

def build_critic_prompt(request: Any, schema_text: str, invalid_sql: str, error_msg: str) -> str:
    """Build the prompt for the critic/self-correction loop."""
    enriched_question = build_enriched_question(
        request.question, 
        request.filters, 
        request.group_columns, 
        request.columns_list
    )
    return f"""You are an Oracle SQL expert. The previous SQL you generated failed with an error.
Please review the schema and the error message, and provide a corrected Oracle SQL statement.

[User Question]
{enriched_question}

【database schema】
{schema_text}

[Previous Invalid SQL]
{invalid_sql}

[Oracle Error Message]
{error_msg}

Return **Oracle SQL only**. No markdown. No explanation.
```sql"""
