from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
import time

class Session(BaseModel):
    """
    Represents a conversation session with metadata and state.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    app_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    state: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def update_state(self, key: str, value: Any):
        self.state[key] = value
        self.updated_at = time.time()

    def update_metadata(self, key: str, value: Any):
        self.metadata[key] = value
        self.updated_at = time.time()
