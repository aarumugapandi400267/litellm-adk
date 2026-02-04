import litellm
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional

from ..vector_store import VectorStore
from ...observability.logger import adk_logger

class PostgresVectorStore(VectorStore):
    """
    PostgreSQL-based vector store implementation using pgvector.
    
    Requires:
    - PostgreSQL database
    - `pgvector` extension installed (`CREATE EXTENSION vector;`)
    - `asyncpg` and `pgvector` python packages
    """
    def __init__(self, 
                 connection_string: str, 
                 table_name: str = "adk_vectors",
                 embedding_model: str = "text-embedding-3-small",
                 vector_dim: int = 1536):
        try:
            import asyncpg
            from pgvector.asyncpg import register_vector
        except ImportError:
            raise ImportError("asyncpg/pgvector not installed. Please run `pip install asyncpg pgvector`.")
        
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_model = embedding_model
        self.vector_dim = vector_dim
        self.pool = None

    async def _ensure_pool(self):
        import asyncpg
        from pgvector.asyncpg import register_vector
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.connection_string)
            async with self.pool.acquire() as conn:
                await register_vector(conn)
                # Ensure extension and table exist
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id UUID PRIMARY KEY,
                        text TEXT,
                        metadata JSONB,
                        embedding vector({self.vector_dim})
                    )
                """)

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LiteLLM."""
        response = await litellm.aembedding(
            model=self.embedding_model,
            input=texts
        )
        return [d["embedding"] for d in response["data"]]

    async def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> List[str]:
        await self._ensure_pool()
        import json
        
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]
        if not metadatas:
            metadatas = [{} for _ in texts]
            
        embeddings = await self._get_embeddings(texts)
        
        records = []
        for i, text in enumerate(texts):
            records.append((
                ids[i],
                text,
                json.dumps(metadatas[i]),
                embeddings[i]
            ))
            
        async with self.pool.acquire() as conn:
            # Upsert logic
            await conn.executemany(f"""
                INSERT INTO {self.table_name} (id, text, metadata, embedding)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
            """, records)
            
        return ids

    async def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        await self._ensure_pool()
        import json
        
        query_embedding = (await self._get_embeddings([query]))[0]
        
        async with self.pool.acquire() as conn:
            # Cosine distance operator (<=>)
            rows = await conn.fetch(f"""
                SELECT id, text, metadata, 1 - (embedding <=> $1) as score
                FROM {self.table_name}
                ORDER BY embedding <=> $1
                LIMIT $2
            """, query_embedding, k)
            
        results = []
        for row in rows:
            results.append({
                "id": str(row["id"]),
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
                "score": float(row["score"])
            })
            
        return results

    async def close(self):
        if self.pool:
            await self.pool.close()
