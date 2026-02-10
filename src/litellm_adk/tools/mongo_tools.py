
import logging
from typing import List, Dict, Any, Optional, Union
import pymongo
from pymongo import MongoClient

logger = logging.getLogger("litellm_adk.tools.mongo")

class MongoTools:
    """
    Manages MongoDB connection, schema (collection) introspection, and query execution.
    """
    def __init__(self, db_url: str, db_name: Optional[str] = None):
        """
        Initialize Mongo Tools.
        Args:
            db_url: MongoDB connection string.
            db_name: Target database name (required if not in URI).
        """
        self.client = MongoClient(db_url)
        # Parse db_name if not provided but in URI? 
        # For simplicity, we assume generic URI, but let's try to get default from client.
        try:
            self.db = self.client.get_database(db_name)
        except Exception:
            # Fallback to 'test' or let it fail
             self.db = self.client.get_database('test')
             
    def get_collection_names(self) -> List[str]:
        """Returns list of collection names."""
        try:
             return self.db.list_collection_names()
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    def get_schema_summary(self, collection_names: Optional[List[str]] = None) -> str:
        """
        Returns a simplified schema summary.
        Args:
             collection_names: specific list of collections.
        """
        try:
            target_colls = collection_names
            if not target_colls:
                target_colls = self.get_collection_names()
            
            schema_str = []
            
            for coll_name in target_colls:
                # Get a sample document to learn structure
                sample = self.db[coll_name].find_one()
                if sample:
                    # Simplify types for LLM
                    # Recursively map types? Or just JSON dump
                    import json
                    # Use default=str for ObjectIds
                    sample_json = json.dumps(sample, default=str, indent=2)
                    # Truncate if too long
                    if len(sample_json) > 500:
                        sample_json = sample_json[:500] + "... (truncated)"
                    
                    schema_str.append(f"Collection '{coll_name}':\n  Sample Document: {sample_json}")
                else:
                    schema_str.append(f"Collection '{coll_name}': (Empty)")
            
            return "\n\n".join(schema_str)
        except Exception as e:
            logger.error(f"Mongo introspection failed: {e}")
            return "Error: Could not retrieve MongoDB schema."

    def execute_mongo_find(self, collection: str, query: Union[Dict[str, Any], str], limit: int = 10) -> str:
        """
        Executes a 'find' query.
        """
        try:
            coll = self.db[collection]
            
            # Flexible Parsing
            if isinstance(query, str):
                import json
                try:
                    query = json.loads(query)
                except json.JSONDecodeError:
                    return "Error: Invalid JSON string for query."
            
            if not isinstance(query, dict):
                 return "Error: Query must be a JSON object (dictionary)."

            cursor = coll.find(query).limit(limit)
            results = list(cursor)
            
            if not results:
                return "No documents found."
            
            import json
            json_res = json.dumps(results, default=str, indent=2)
            
            # Smart Truncation
            if len(json_res) > 3000:
                return f"Found {len(results)} docs. Response truncated:\n{json_res[:3000]}\n...(Use aggregations or filters)"
            
            return json_res
        except Exception as e:
            return f"Mongo Execution Error: {str(e)}"
            
    def execute_mongo_aggregate(self, collection: str, pipeline: Union[List[Dict[str, Any]], str]) -> str:
        """
        Executes an aggregation pipeline.
        """
        try:
            coll = self.db[collection]
            
            # Flexible Parsing
            if isinstance(pipeline, str):
                import json
                try:
                    pipeline = json.loads(pipeline)
                except json.JSONDecodeError:
                    return "Error: Invalid JSON string for pipeline."
            
            if not isinstance(pipeline, list):
                return "Error: Pipeline must be a JSON array (list)."

            # Safety: Restrict $out, $merge?
            # Basic check
            for stage in pipeline:
                if not isinstance(stage, dict):
                    continue
                for k in stage.keys():
                    if k in ["$out", "$merge"]:
                        return "Security Error: Aggregation stages '$out' and '$merge' are prohibited."
            
            results = list(coll.aggregate(pipeline))
            
            if not results:
                 return "No results."
                 
            import json
            json_res = json.dumps(results, default=str, indent=2)
            
            # Smart Truncation
            if len(json_res) > 3000:
                return f"Aggregation success. {len(results)} docs. Response truncated:\n{json_res[:3000]}\n...(Use strict filters)"
            
            return json_res
        except Exception as e:
             return f"Mongo Aggregation Error: {str(e)}"
