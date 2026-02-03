import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel, create_model
from ..observability.logger import adk_logger

class ToolRegistry:
    """
    Registry for managing tools that can be called by agents.
    """
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name_or_func: Any = None):
        """
        Decorator to register a function as a tool.
        Supports both @tool and @tool(name="...")
        """
        if callable(name_or_func):
            self._register_function(name_or_func)
            return name_or_func

        def decorator(func: Callable):
            self._register_function(func, name_or_func)
            return func
        return decorator

    def _register_function(self, func: Callable, name: Optional[str] = None) -> Dict[str, Any]:
        """Internal helper to register a function and return its definition."""
        tool_name = name or func.__name__
        description = func.__doc__ or f"Tool: {tool_name}"
        
        sig = inspect.signature(func)
        parameters = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self": continue
            param_type = "string"
            if param.annotation == int: param_type = "integer"
            elif param.annotation == float: param_type = "number"
            elif param.annotation == bool: param_type = "boolean"
            
            parameters[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}"
            }

        definition = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description.strip(),
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": [p.name for p in sig.parameters.values() if p.default == inspect.Parameter.empty and p.name != "self"]
                }
            }
        }
        
        self._tools[tool_name] = {
            "name": tool_name,
            "func": func,
            "definition": definition
        }
        adk_logger.debug(f"Registered tool: {tool_name}")
        return definition

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Returns list of tool definitions in OpenAI format.
        """
        return [t["definition"] for t in self._tools.values()]

    def execute(self, name: str, **kwargs) -> Any:
        """
        Executes a registered tool by name with keyword arguments.
        """
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in registry.")
            
        adk_logger.info(f"Executing tool: {name} with args: {kwargs}")
        return self._tools[name]["func"](**kwargs)

# Global tool registry
tool_registry = ToolRegistry()
tool = tool_registry.register
