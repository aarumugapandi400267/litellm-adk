from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Generator, AsyncGenerator

class BaseAgent(ABC):
    """
    Abstract Base Class for all agents.
    """
    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> Any:
        pass

    @abstractmethod
    async def ainvoke(self, prompt: str, **kwargs) -> Any:
        pass
