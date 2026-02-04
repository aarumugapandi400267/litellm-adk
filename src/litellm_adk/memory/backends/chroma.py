import litellm
import uuid
import logging
from typing import List, Dict, Any, Optional

from ..vector_store import VectorStore
from ...observability.logger import adk_logger

class ChromaVectorStore(VectorStore):
    """
    ChromaDB-based vector store implementation.
    Uses LiteLLM for embedding generation to ensure compatibility.
    """
    def __init__(self, collection_name: str = "adk_memory", embedding_model: str = "text-embedding-3-small", persist_path: str = "./chroma_db"):
        try:
            import chromadb
            # Use persistent client by default
            self.client = chromadb.PersistentClient(path=persist_path)
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except ImportError:
            raise ImportError("ChromaDB not installed. Please run `pip install chromadb`.")
        
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
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return ids

    async def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        query_embedding = (await self._get_embeddings([query]))[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Parse Chroma result format into standard Dict
        parsed_results = []
        if results["ids"]:
            count = len(results["ids"][0])
            for i in range(count):
                parsed_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": results["distances"][0][i] if results["distances"] else 0.0
                })
        
        return parsed_results
