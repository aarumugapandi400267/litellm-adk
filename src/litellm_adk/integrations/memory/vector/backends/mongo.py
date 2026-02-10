import litellm
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional

from .....core.vector_store import VectorStore
from .....observability.logger import adk_logger

class MongoVectorStore(VectorStore):
    """
    MongoDB-based vector store implementation using Atlas Vector Search.
    
    Requires a MongoDB Atlas cluster with a vector search index configured.
    """
    def __init__(self, 
                 connection_string: str, 
                 database_name: str = "adk_memory", 
                 collection_name: str = "vectors",
                 index_name: str = "vector_index",
                 embedding_model: str = "text-embedding-3-small"):
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            self.client = AsyncIOMotorClient(connection_string)
            self.db = self.client[database_name]
            self.collection = self.db[collection_name]
            self.index_name = index_name
        except ImportError:
            raise ImportError("Motor not installed. Please run `pip install motor`.")
        
        self.embedding_model = embedding_model

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LiteLLM."""
        response = await litellm.aembedding(
            model=self.embedding_model,
            input=texts
        )
        return [d["embedding"] for d in response["data"]]

    async def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> List[str]:
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]
        if not metadatas:
            metadatas = [{} for _ in texts]
            
        embeddings = await self._get_embeddings(texts)
        
        documents = []
        for i, text in enumerate(texts):
            doc = {
                "id": ids[i],
                "text": text,
                "embedding": embeddings[i],
                "metadata": metadatas[i]
            }
            documents.append(doc)
            
        if documents:
            await self.collection.insert_many(documents)
            
        return ids

    async def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        query_embedding = (await self._get_embeddings([query]))[0]
        
        # Atlas Vector Search Aggregation Pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": k * 10,
                    "limit": k
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "id": 1,
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        cursor = self.collection.aggregate(pipeline)
        results = await cursor.to_list(length=k)
        
        # Standardize output
        return list(results)
