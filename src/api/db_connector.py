import oracledb
import os
from typing import Tuple, Optional, Any, Dict
from .config import settings

# Initialize Oracle Client if lib_dir is provided
if settings.ORACLE_LIB_DIR and os.path.exists(settings.ORACLE_LIB_DIR):
    try:
        oracledb.init_oracle_client(lib_dir=settings.ORACLE_LIB_DIR)
    except Exception as e:
        print(f"Warning: Could not initialize Oracle client: {e}. Using Thin mode.")

def validate_sql_with_oracle(sql: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate SQL by attempting to EXPLAIN PLAN or execute a dry run.
    Returns (is_valid, error_message, metadata).
    """
    connection = None
    try:
        connection = oracledb.connect(
            user=settings.ORACLE_USER,
            password=settings.ORACLE_PASSWORD,
            dsn=settings.ORACLE_DSN
        )
        connection.call_timeout = settings.DB_TIMEOUT * 1000 # in milliseconds
        cursor = connection.cursor()
        
        # Use EXPLAIN PLAN to validate syntax and schema without executing
        # This is safer than executing the actual query
        explain_query = f"EXPLAIN PLAN FOR {sql.rstrip(';')}"
        cursor.execute(explain_query)
        
        cursor.close()
        return True, None, {"status": "parsed_successfully"}
        
    except oracledb.Error as e:
        error_obj = e.args[0]
        return False, str(error_obj.message if hasattr(error_obj, 'message') else error_obj), None
    except Exception as e:
        return False, str(e), None
    finally:
        if connection:
            connection.close()

def execute_oracle_query(sql: str) -> Tuple[Optional[list], Optional[str]]:
    """Execute SQL and return results or error."""
    connection = None
    try:
        connection = oracledb.connect(
            user=settings.ORACLE_USER,
            password=settings.ORACLE_PASSWORD,
            dsn=settings.ORACLE_DSN
        )
        connection.call_timeout = settings.DB_TIMEOUT * 1000 # in milliseconds
        cursor = connection.cursor()
        cursor.execute(sql)
        # Fetch first 100 rows for brevity
        results = cursor.fetchmany(100)
        cursor.close()
        return results, None
    except Exception as e:
        return None, str(e)
    finally:
        if connection:
            connection.close()
