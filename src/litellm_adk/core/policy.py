from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class PolicyRule:
    tool_name: str
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    description: str = ""

class PolicyEngine:
    """
    Evaluates whether a tool call should trigger HITL based on rules.
    """
    def __init__(self):
        self.rules: List[PolicyRule] = []

    def add_rule(self, tool_name: str, condition: Optional[Callable[[Dict[str, Any]], bool]] = None, description: str = ""):
        self.rules.append(PolicyRule(tool_name=tool_name, condition=condition, description=description or f"Rule for {tool_name}"))

    def should_require_approval(self, tool_name: str, args: Dict[str, Any]) -> bool:
        for rule in self.rules:
            if rule.tool_name == tool_name:
                if rule.condition is None or rule.condition(args):
                    return True
        return False
