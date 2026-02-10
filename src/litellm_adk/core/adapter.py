from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional

class DatabaseAdapter(ABC):
    """
    Abstract Base Class for Database Adapters.
    """
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adapter with connection configuration.
        config: {"url": "...", "username": "...", "password": "...", "host": "...", "port": ...}
        """
        pass

    @abstractmethod
    def get_table_names(self) -> List[str]:
        """Return a list of all table/collection names."""
        pass

    @abstractmethod
    def get_schema_summary(self, table_names: Optional[List[str]] = None) -> str:
        """Return schema definition (DDL or Sample Docs) for requested tables."""
        pass

    @abstractmethod
    def get_tools(self) -> List[Any]:
        """Return a list of callable tools (functions) to be bound to the Agent."""
        pass

    def set_result_callback(self, callback: callable):
        """Sets a callback function(data: Any) to be called when a query executes."""
        self.result_callback = callback
    
    @abstractmethod
    def get_system_prompt_template(self) -> str:
        """Return the database-specific instructions for the system prompt."""
        pass
