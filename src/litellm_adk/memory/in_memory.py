from typing import List, Dict, Any
from .base import BaseMemory

class InMemoryMemory(BaseMemory):
    """
    Standard in-memory store for conversation history and session metadata.
    """
    def __init__(self):
        self._storage: Dict[str, List[Dict[str, Any]]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        return self._storage.get(session_id, []).copy()

    def add_message(self, session_id: str, message: Dict[str, Any]):
        if session_id not in self._storage:
            self._storage[session_id] = []
        self._storage[session_id].append(message)

    def add_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        if session_id not in self._storage:
            self._storage[session_id] = []
        self._storage[session_id].extend(messages)

    def clear(self, session_id: str):
        if session_id in self._storage:
            self._storage[session_id] = []
        if session_id in self._metadata:
            self._metadata[session_id] = {}

    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        return self._metadata.get(session_id, {}).copy()

    def save_session_metadata(self, session_id: str, metadata: Dict[str, Any]):
        self._metadata[session_id] = metadata.copy()
