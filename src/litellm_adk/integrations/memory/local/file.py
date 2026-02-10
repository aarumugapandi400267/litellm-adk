import json
import os
from typing import List, Dict, Any, Optional
from ....core.memory import BaseMemory

class FileMemory(BaseMemory):
    """
    JSON file-based persistence for conversation history and session metadata.
    """
    def __init__(self, file_path: str = "conversations.json"):
        self.file_path = file_path
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    # Support legacy format (where values were just lists)
                    for k, v in data.items():
                        if isinstance(v, list):
                            self._cache[k] = {"messages": v, "metadata": {}}
                        else:
                            self._cache[k] = v
                except json.JSONDecodeError:
                    self._cache = {}
        else:
            self._cache = {}

    def _save(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2, ensure_ascii=False)

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        return self._cache.get(session_id, {}).get("messages", []).copy()

    def add_message(self, session_id: str, message: Dict[str, Any]):
        if session_id not in self._cache:
            self._cache[session_id] = {"messages": [], "metadata": {}}
        self._cache[session_id]["messages"].append(message)
        self._save()

    def add_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        if session_id not in self._cache:
            self._cache[session_id] = {"messages": [], "metadata": {}}
        self._cache[session_id]["messages"].extend(messages)
        self._save()

    def clear(self, session_id: str):
        if session_id in self._cache:
            self._cache[session_id] = {"messages": [], "metadata": {}}
            self._save()

    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        return self._cache.get(session_id, {}).get("metadata", {}).copy()

    def save_session_metadata(self, session_id: str, metadata: Dict[str, Any]):
        if session_id not in self._cache:
            self._cache[session_id] = {"messages": [], "metadata": {}}
        self._cache[session_id]["metadata"] = metadata
        self._save()
