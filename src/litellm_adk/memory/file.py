import json
import os
from typing import List, Dict, Any, Optional
from .base import BaseMemory

class FileMemory(BaseMemory):
    """
    JSON file-based persistence for conversation history.
    """
    def __init__(self, file_path: str = "conversations.json"):
        self.file_path = file_path
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                try:
                    self._cache = json.load(f)
                except json.JSONDecodeError:
                    self._cache = {}
        else:
            self._cache = {}

    def _save(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2, ensure_ascii=False)

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        return self._cache.get(session_id, []).copy()

    def add_message(self, session_id: str, message: Dict[str, Any]):
        if session_id not in self._cache:
            self._cache[session_id] = []
        self._cache[session_id].append(message)
        self._save()

    def add_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        if session_id not in self._cache:
            self._cache[session_id] = []
        self._cache[session_id].extend(messages)
        self._save()

    def clear(self, session_id: str):
        if session_id in self._cache:
            self._cache[session_id] = []
            self._save()
