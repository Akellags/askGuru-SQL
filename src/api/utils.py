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
        lines.append(f"Table: {table['table_name']}")
        if "notes" in table:
            lines.append(f"Description: {table['notes']}")
        lines.append("Columns:")
        for col in table['columns']:
            col_line = f"- {col['name']} ({col['type']}): {col.get('description', '')}"
            if col.get('is_primary'):
                col_line += " [PRIMARY KEY]"
            lines.append(col_line)
        
        if table.get("foreign_keys"):
            lines.append("Foreign Keys:")
            for fk in table['foreign_keys']:
                lines.append(f"- {fk['column']} -> {fk['ref_table']}.{fk['ref_column']}")
        lines.append("") # Spacer
    
    return "\n".join(lines)

def get_filtered_schema(mschema: List[Dict[str, Any]], requested_tables: Optional[List[str]] = None) -> str:
    """Get formatted schema text, optionally filtered by table names."""
    if not requested_tables:
        return format_mschema_text(mschema)
    
    filtered = [t for t in mschema if t['table_name'].lower() in [rt.lower() for rt in requested_tables]]
    return format_mschema_text(filtered)

def build_llama_prompt(question: str, schema_text: str) -> str:
    """Build the prompt for LLaMA-3.1 model."""
    return f"""You are a SQL expert. You need to read and understand the following 【database schema】 description and generate a valid Oracle SQL statement to answer the [User Question].

Return **Oracle SQL only**. No markdown. No explanation.

【database schema】
{schema_text}

[User Question]
{question}

```sql"""

def build_sqlcoder_prompt(question: str, schema_text: str) -> str:
    """Build the prompt for SQLCoder-70B model."""
    return f"""### Task
Generate a SQL query to answer the following question: `{question}`

### Database Schema
{schema_text}

### SQL Query
"""

def build_critic_prompt(question: str, schema_text: str, invalid_sql: str, error_msg: str) -> str:
    """Build the prompt for the critic/self-correction loop."""
    return f"""You are an Oracle SQL expert. The previous SQL you generated failed with an error.
Please review the schema and the error message, and provide a corrected Oracle SQL statement.

[User Question]
{question}

【database schema】
{schema_text}

[Previous Invalid SQL]
{invalid_sql}

[Oracle Error Message]
{error_msg}

Return **Oracle SQL only**. No markdown. No explanation.
```sql"""
