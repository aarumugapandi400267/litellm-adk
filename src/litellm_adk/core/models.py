from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    EXPIRED = "expired"

class ApprovalRequest(BaseModel):
    id: str = Field(..., description="Unique tool call ID")
    session_id: str
    tool_name: str
    original_args: Dict[str, Any]
    modified_args: Optional[Dict[str, Any]] = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    requester: str = "agent"
    reviewer: Optional[str] = None
    reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

class ApprovalAuditEntry(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str
    session_id: str
    action: str  # e.g., "created", "approved", "rejected", "modified"
    actor: str
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentResponse(BaseModel):
    """
    Structured response from an agent invocation.
    Provies both the final answer and the full trace for UI flexibility.
    """
    content: str = Field(..., description="The final text response from the agent.")
    accumulated_content: str = Field(..., description="The full concatenated text including intermediate thoughts.")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Tools executed during this turn.")
    session_id: str
    
    def __str__(self):
        """Default to the full accumulated content for convenient printing."""
        return self.accumulated_content
