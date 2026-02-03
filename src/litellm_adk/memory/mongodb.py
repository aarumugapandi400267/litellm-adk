from typing import List, Dict, Any, Optional
from .base import BaseMemory
import pymongo

class MongoDBMemory(BaseMemory):
    """
    MongoDB-based persistence for conversation history and session metadata.
    """
    def __init__(
        self, 
        connection_string: str = "mongodb://localhost:27017/",
        database_name: str = "litellm_adk",
        collection_name: str = "conversations"
    ):
        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        # Create index on session_id for faster lookups
        self.collection.create_index("session_id", unique=True)

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        doc = self.collection.find_one({"session_id": session_id})
        if doc:
            return list(doc.get("messages", []))
        return []

    def add_message(self, session_id: str, message: Dict[str, Any]):
        self.collection.update_one(
            {"session_id": session_id},
            {"$push": {"messages": message}},
            upsert=True
        )

    def add_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        self.collection.update_one(
            {"session_id": session_id},
            {"$push": {"messages": {"$each": messages}}},
            upsert=True
        )

    def clear(self, session_id: str):
        self.collection.update_one(
            {"session_id": session_id},
            {"$set": {"messages": [], "metadata": {}}}
        )

    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        doc = self.collection.find_one({"session_id": session_id})
        if doc:
            return doc.get("metadata", {})
        return {}

    def save_session_metadata(self, session_id: str, metadata: Dict[str, Any]):
        self.collection.update_one(
            {"session_id": session_id},
            {"$set": {"metadata": metadata}},
            upsert=True
        )
