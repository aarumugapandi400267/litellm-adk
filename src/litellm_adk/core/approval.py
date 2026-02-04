import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from .models import ApprovalRequest, ApprovalStatus, ApprovalAuditEntry
from ..observability.logger import adk_logger

class ApprovalManager:
    """
    Manages the lifecycle of tool call approvals.
    Uses a simple file-based store by default.
    """
    def __init__(self, storage_path: str = "approvals.json"):
        self.storage_path = storage_path
        self._requests: Dict[str, ApprovalRequest] = {}
        self._audit_trail: List[ApprovalAuditEntry] = []
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    self._requests = {k: ApprovalRequest(**v) for k, v in data.get("requests", {}).items()}
                    # Audit trail is likely too large for a single JSON in production, 
                    # but fine for ADK architecture
            except Exception as e:
                adk_logger.error(f"Failed to load approvals: {e}")

    def _save(self):
        try:
            with open(self.storage_path, "w") as f:
                data = {
                    "requests": {k: v.model_dump(mode='json') for k, v in self._requests.items()}
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            adk_logger.error(f"Failed to save approvals: {e}")

    def create_request(self, id: str, session_id: str, tool_name: str, args: Dict[str, Any]) -> ApprovalRequest:
        request = ApprovalRequest(
            id=id,
            session_id=session_id,
            tool_name=tool_name,
            original_args=args
        )
        self._requests[id] = request
        self._log_audit(id, session_id, "created", "agent")
        self._save()
        return request

    def get_request(self, id: str) -> Optional[ApprovalRequest]:
        return self._requests.get(id)

    def submit_decision(self, id: str, status: ApprovalStatus, reviewer: str = "human", reason: Optional[str] = None, modified_args: Optional[Dict[str, Any]] = None):
        request = self.get_request(id)
        if not request:
            raise ValueError(f"Approval request {id} not found.")

        request.status = status
        request.reviewer = reviewer
        request.reason = reason
        if modified_args:
            request.modified_args = modified_args
        
        self._log_audit(id, request.session_id, status.value, reviewer, reason, {"modified_args": modified_args} if modified_args else {})
        self._save()

    def _log_audit(self, request_id: str, session_id: str, action: str, actor: str, reason: Optional[str] = None, metadata: Dict[str, Any] = None):
        entry = ApprovalAuditEntry(
            request_id=request_id,
            session_id=session_id,
            action=action,
            actor=actor,
            reason=reason,
            metadata=metadata or {}
        )
        # In a real system, this goes to a sequence-only log/db
        adk_logger.info(f"AUDIT - HITL: {action} on {request_id} by {actor} (reason: {reason})")

    def get_effective_args(self, id: str, default_args: Dict[str, Any] = None) -> Dict[str, Any]:
        """Returns modified args if present, otherwise original (or default)."""
        request = self.get_request(id)
        if not request:
            return default_args or {}
        return request.modified_args if request.modified_args is not None else request.original_args
