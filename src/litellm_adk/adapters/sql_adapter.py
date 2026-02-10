from typing import List, Dict, Any, Union, Optional
import logging
from .base import DatabaseAdapter
from ..tools.sql_tools import SQLTools

logger = logging.getLogger("litellm_adk.adapters.sql")

class SQLAdapter(DatabaseAdapter):
    """
    Adapter for SQL Relational Databases (Postgres, MySQL, Sqlite, etc).
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Config Schema:
        - url: Full connection URI (Optional if host/user/pass provided)
        - host, port, username, password, database (Optional, will synthesize URI)
        - schema_config: Dict (Optional) e.g., {'exclude_tables': ['secrets']}
        """
        self.config = config
        self.db_url = config.get("url")
        
        # Synthesize URL if not provided but components are
        if not self.db_url:
            user = config.get("username", "")
            password = config.get("password", "")
            host = config.get("host", "localhost")
            port = config.get("port", 5432)
            dbname = config.get("database", "postgres")
            driver = config.get("driver", "postgresql") # or mysql+pymysql
            
            auth = f"{user}:{password}@" if user else ""
            self.db_url = f"{driver}://{auth}{host}:{port}/{dbname}"
            
        self.tools = SQLTools(self.db_url, config.get("schema_config"))
        logger.info(f"Initialized SQL Adapter for {self.db_url.split('@')[-1]}")

    def get_table_names(self) -> List[str]:
        return self.tools.get_table_names()

    def get_schema_summary(self, table_names: Optional[List[str]] = None) -> str:
        return self.tools.get_schema_summary(table_names)

    def get_tools(self) -> List[Any]:
        # Wrap tool method to be bindable
        def execute_sql(query: str):
            """Execute a SQL query. Returns results as JSON."""
            result_str = self.tools.execute_sql_tool(query)
            
            # If callback registered, try to capture the data.
            # But wait, execute_sql_tool returns a STRING.
            # Ideally, the TOOL should return (data, string).
            # For now, let's just callback with the string (JSON) or try to decode it.
            if hasattr(self, 'result_callback') and self.result_callback:
                 import json
                 try:
                     # Remove the "Query executed successfully..." prefix if present
                     json_part = result_str
                     if "\n" in result_str:
                         parts = result_str.split("\n", 1)
                         if parts[0].startswith("Query executed"):
                             json_part = parts[1]
                             # Remove truncation suffix if present
                             if "... (Result truncated" in json_part:
                                 json_part = json_part.split("\n... (Result truncated")[0]
                     
                     data = json.loads(json_part)
                     self.result_callback(data)
                     
                     # Blind Mode: Return summary only
                     count = len(data) if isinstance(data, list) else 1
                     return f"Query executed successfully. Found {count} rows. Data hidden from LLM and sent to UI."
                 except:
                     pass # Failed to parse JSON or just an error message string
            
            return result_str
        
        return [execute_sql]

    def get_system_prompt_template(self) -> str:
        return """
1. Generate a valid SQL query to answer the user's question.
2. Select relevant tables from the schema.
3. Use the 'execute_sql' tool (which you can call) to run the query.
4. Only use SELECT statements. Data modification is strictly prohibited.
"""
