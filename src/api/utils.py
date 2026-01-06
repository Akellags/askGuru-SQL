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

def build_llama_prompt(request: Any, mschema: List[Dict[str, Any]], schema_text: str) -> str:
    """Build the prompt for LLaMA-3.1 model matching test_script_with_full_prompt.py format."""
    enriched_question = build_enriched_question(
        request.question, 
        request.filters, 
        request.group_columns, 
        request.columns_list
    )
    
    join_hints = get_join_hints(mschema, request.tables or [])
    
    return f"""You are a Oracle EBS expert. Generate executable SQL based on the user's question.
Only output SQL query. Do not invent columns - use only those in the schema.
CRITICAL: Columns marked with **[ESSENTIAL]** are mandatory for proper joins and aggregations - prioritize them.

[User Question]
{enriched_question}

[Database Schema]
{schema_text}
{join_hints}

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
