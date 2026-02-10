from typing import List, Dict, Any, Optional
import logging
from ..sql.adapter import SQLAdapter

logger = logging.getLogger("litellm_adk.integrations.database.mysql")

class MySQLAdapter(SQLAdapter):
    """
    Specialized Adapter for MySQL Databases.
    Extends SQLAdapter with MySQL-specific defaults and prompt instructions.
    """
    def __init__(self, config: Dict[str, Any]):
        # Default to mysql+pymysql if no driver/url provided
        if not config.get("url") and not config.get("driver"):
             config["driver"] = "mysql+pymysql"
             
        if not config.get("url") and "port" not in config:
            config["port"] = 3306
            
        if not config.get("url") and "database" not in config:
            # MySQL usually needs a database name in the path
            config["database"] = config.get("dbname", "mysql")
            
        super().__init__(config)
        logger.info(f"Initialized MySQL Adapter (Driver: {config.get('driver')})")

    def get_system_prompt_template(self) -> str:
        """MySQL-specific prompt instructions."""
        return """
1. Generate valid MySQL-compliant SQL.
2. If you are unsure of the schema for a table, use 'inspect_table' first.
3. IMPORTANT: Use backticks (`) for all table and column names (e.g., `users`.`id`) to avoid conflicts with MySQL reserved keywords.
4. Use 'LIMIT' for all queries unless the user explicitly asks for all records.
5. For date comparisons, use standard MySQL date functions or strings 'YYYY-MM-DD'.
6. Only use SELECT statements. Do NOT attempt to INSERT, UPDATE, or DELETE.
7. Use the 'execute_sql' tool provided to run your queries.
"""
