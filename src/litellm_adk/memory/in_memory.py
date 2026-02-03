from typing import List, Dict, Any
from .base import BaseMemory

class InMemoryMemory(BaseMemory):
    """
    Standard in-memory store for conversation history.
    """
    def __init__(self):
        self._storage: Dict[str, List[Dict[str, Any]]] = {}

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
