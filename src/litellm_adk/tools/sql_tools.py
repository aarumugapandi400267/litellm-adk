import json
import logging
from typing import List, Dict, Any, Optional, Union
import sqlalchemy
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
import sqlparse

logger = logging.getLogger("litellm_adk.tools.sql")

class SQLTools:
    """
    Manages database connection, schema introspection, and secure SQL execution.
    Designed to be used by NL2SQLAgent.
    """
    def __init__(self, db_url: str, schema_config: Optional[Dict[str, Any]] = None):
        """
        Initialize SQL Tools.
        
        Args:
            db_url: Database connection string (SQLAlchemy format).
            schema_config: Optional config to filter tables.
                           {'include_tables': ['users'], 'exclude_tables': ['secrets']}
        """
        self.db_url = db_url
        self.engine = create_engine(db_url) # Synchronous engine (offloaded by Agent)
        self.schema_config = schema_config or {}
        
    def get_table_names(self) -> List[str]:
        """Returns a list of all table names."""
        try:
            inspector = inspect(self.engine)
            all_tables = inspector.get_table_names()
            
            # Apply Config Filters
            include = self.schema_config.get('include_tables')
            exclude = self.schema_config.get('exclude_tables', [])
            
            if include:
                all_tables = [t for t in all_tables if t in include]
            
            return [t for t in all_tables if t not in exclude]
        except Exception as e:
            logger.error(f"Error fetching table names: {e}")
            return []

    def get_schema_summary(self, table_names: Optional[List[str]] = None) -> str:
        """
        Returns a simplified schema summary (DDL-like) for the LLM prompt.
        Args:
            table_names: specific list of tables to inspect. If None, checks ALL (filtered) tables.
        """
        try:
            inspector = inspect(self.engine)
            
            target_tables = table_names
            if not target_tables:
                target_tables = self.get_table_names()
            
            schema_str = []
            for table in target_tables:
                # Basic check to avoid errors if table doesn't exist
                if not inspector.has_table(table):
                     continue

                columns = inspector.get_columns(table)
                # Format: Table Name (col1: type, col2: type)
                col_defs = [f"{c['name']}: {c['type']}" for c in columns]
                schema_str.append(f"Table '{table}':\n  Columns: {', '.join(col_defs)}")
                
                # Add Foreign Keys if useful
                try:
                    fks = inspector.get_foreign_keys(table)
                    if fks:
                        fk_strs = [f"{fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}" for fk in fks]
                        schema_str.append(f"  Foreign Keys: {', '.join(fk_strs)}")
                except Exception:
                    pass
            
            return "\n\n".join(schema_str)
        except Exception as e:
            logger.error(f"Schema introspection failed: {e}")
            return "Error: Could not retrieve database schema."

    def validate_sql(self, query: str) -> Union[bool, str]:
        """
        Validates SQL for safety.
        Returns True if safe, or an error string if blocked.
        """
        try:
            parsed = sqlparse.parse(query)
            for statement in parsed:
                # Check statement type
                type_ = statement.get_type().upper()
                if type_ not in ['SELECT', 'DESCRIBE', 'EXPLAIN', 'SHOW']:
                    return f"Security Error: Only SELECT queries are allowed. Blocked command type: {type_}"
        except Exception as e:
            return f"Validation Error: {e}"
        return True

    def execute_sql_tool(self, query: str) -> str:
        """
        The actual Tool function exposed to the LLM.
        Executes the query and returns JSON results or Error message.
        """
        query = query.strip()
        
        # 1. Validate
        validation = self.validate_sql(query)
        if validation is not True:
            return str(validation)
        
        # 2. Execute
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                
                # Fetch results (limit to avoid token overflow)
                # We can't use LIMIT in SQL easily without parsing/rewriting, 
                # so we fetchmany.
                rows = result.fetchmany(100)
                keys = result.keys()
                
                data = [dict(zip(keys, row)) for row in rows]
                
                if not data:
                    return "Query executed successfully in 0.0s. No rows returned."
                
                # Convert to string (JSON is good for LLM)
                import json
                json_res = json.dumps(data, default=str, indent=2)
                
                # Smart Truncation: Hard limit on characters to protect Context Window
                start_marker = f"Query executed successfully. Returned {len(data)} rows.\n"
                if len(json_res) > 3000:
                    return start_marker + json_res[:3000] + "\n... (Result truncated. Please refine query with LIMIT or Aggregations)"
                
                return start_marker + json_res
                
        except SQLAlchemyError as e:
            # Return the specific DB error so the LLM can fix it
            return f"Database Error: {str(e.__cause__) or str(e)}"
        except Exception as e:
            return f"Execution Error: {str(e)}"
