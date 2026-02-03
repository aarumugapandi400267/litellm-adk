from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseMemory(ABC):
    """
    Abstract Base Class for memory persistence.
    """
    
    @abstractmethod
    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all messages for a given session."""
        pass

    @abstractmethod
    def add_message(self, session_id: str, message: Dict[str, Any]):
        """Add a single message to a session."""
        pass

    @abstractmethod
    def add_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        """Add multiple messages to a session."""
        pass

    @abstractmethod
    def clear(self, session_id: str):
        """Clear history for a session."""
        pass

    @abstractmethod
    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """Retrieve metadata/state for a given session."""
        pass

    @abstractmethod
    def save_session_metadata(self, session_id: str, metadata: Dict[str, Any]):
        """Save/Update metadata/state for a given session."""
        pass
