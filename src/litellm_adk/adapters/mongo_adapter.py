from typing import List, Dict, Any, Union, Optional
import logging
from .base import DatabaseAdapter
from ..tools.mongo_tools import MongoTools

logger = logging.getLogger("litellm_adk.adapters.mongo")

class MongoAdapter(DatabaseAdapter):
    """
    Adapter for MongoDB.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Config Schema:
        - url: Full connection URI (Optional if host/user/pass provided)
        - host, port, username, password, database (Optional, will synthesize URI)
        """
        self.config = config
        self.db_url = config.get("url")
        
        # Synthesize URL if not provided.
        if not self.db_url:
            user = config.get("username", "")
            password = config.get("password", "")
            host = config.get("host", "localhost")
            port = config.get("port", 27017)
            dbname = config.get("database", "admin")
            auth = f"{user}:{password}@" if user else ""
            self.db_url = f"mongodb://{auth}{host}:{port}/{dbname}"
            
        self.tools = MongoTools(self.db_url)
        logger.info(f"Initialized MongoDB Adapter for {self.db_url.split('@')[-1]}")

    def get_table_names(self) -> List[str]:
        return self.tools.get_collection_names()

    def get_schema_summary(self, table_names: Optional[List[str]] = None) -> str:
        return self.tools.get_schema_summary(table_names)

    def get_tools(self) -> List[Any]:
        def mongo_find(collection: str, query: Union[Dict[str, Any], str], limit: int = 10):
            """Find documents in a MongoDB collection. query is a JSON dictionary."""
            res_str = self.tools.execute_mongo_find(collection, query, limit)
            callback_success = self._try_callback(res_str)
            
            if callback_success and self.result_callback: 
                # Blind Mode: Return summary only
                import json
                try:
                    data = json.loads(res_str)
                    count = len(data) if isinstance(data, list) else 1
                    return f"Executed Successfully. Found {count} documents. Data hidden from LLM and sent to UI."
                except:
                    return "Executed Successfully. Data sent to UI."
            
            return res_str
        
        def mongo_aggregate(collection: str, pipeline: Union[List[Dict[str, Any]], str]):
            """Execute an aggregation pipeline on a MongoDB collection. pipeline is a list of stages."""
            res_str = self.tools.execute_mongo_aggregate(collection, pipeline)
            callback_success = self._try_callback(res_str)
            
            if callback_success and self.result_callback:
                 import json
                 try:
                     data = json.loads(res_str)
                     count = len(data) if isinstance(data, list) else 1
                     return f"Executed Successfully. Found {count} documents via Aggregation. Data hidden from LLM and sent to UI."
                 except:
                     return "Executed Successfully. Data sent to UI."

            return res_str
            
        return [mongo_find, mongo_aggregate]

    def _try_callback(self, res_str: str) -> bool:
        if hasattr(self, 'result_callback') and self.result_callback:
             import json
             try:
                 clean_json = res_str
                 if "Response truncated" in res_str:
                     # Parse out the JSON part between the header and truncate marker
                     # Header: "Found N docs. Response truncated:\n"
                     # Trailer: "\n...(Use aggregations or filters)"
                     parts = res_str.split("Response truncated:\n")
                     if len(parts) > 1:
                         clean_json = parts[1].split("\n...")[0]
                 
                 data = json.loads(clean_json)
                 self.result_callback(data)
                 return True
             except Exception:
                 pass
        return False

    def get_system_prompt_template(self) -> str:
        return """
1. Analyze the user's request and the Available Tables/Schema.
2. Select relevant collections/tables.
3. Use 'mongo_find' for simple queries or 'mongo_aggregate' for complex analysis.
4. Queries must be valid JSON objects/pipelines.
5. Security: 'mongo_aggregate' cannot use $out or $merge.
"""
