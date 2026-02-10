import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Generator, AsyncGenerator

from .memory import BaseMemory
from .vector_store import VectorStore

class BaseAgent(ABC):
    """
    Abstract Base Class for all agents.
    Provides standard interface for memory, tools, and vector store.
    """
    def __init__(
        self,
        model: str,
        system_prompt: str = "You are a helpful assistant.",
        memory: Optional[BaseMemory] = None,
        vector_store: Optional['VectorStore'] = None,
        **kwargs
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.memory = memory
        self.vector_store = vector_store
        self.kwargs = kwargs

    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> Any:
        pass

    @abstractmethod
    async def ainvoke(self, prompt: str, **kwargs) -> Any:
        pass
