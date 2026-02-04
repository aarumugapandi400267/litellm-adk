from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorStore(ABC):
    """
    Abstract Base Class for Vector Store backends.
    """
    
    @abstractmethod
    async def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> List[str]:
        """Add texts to the vector store."""
        pass

    @abstractmethod
    async def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Search for similar texts.
        Returns a list of dicts with 'text', 'metadata', and 'score'.
        """
        pass
