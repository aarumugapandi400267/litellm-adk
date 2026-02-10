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

    from typing import Union
    def register(self, name_or_func: Any = None, requires_approval: Union[bool, Callable[[Dict[str, Any]], bool]] = False):
        """
        Decorator to register a function as a tool.
        Supports both @tool and @tool(name="...", requires_approval=True)
        """
        if callable(name_or_func):
            self._register_function(name_or_func, requires_approval=requires_approval)
            return name_or_func

        def decorator(func: Callable):
            self._register_function(func, name_or_func, requires_approval=requires_approval)
            return func
        return decorator

    def _register_function(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None, requires_approval: Union[bool, Callable[[Dict[str, Any]], bool]] = False) -> Dict[str, Any]:
        """Internal helper to register a function and return its definition."""
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"
        
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
                "description": tool_description.strip(),
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": [p.name for p in sig.parameters.values() if p.default == inspect.Parameter.empty and p.name != "self"]
                }
            }
        }
        
        # If already registered and no explicit approval flag provided, keep the existing one
        existing = self._tools.get(tool_name)
        final_approval = requires_approval
        if existing and requires_approval is False:
             final_approval = existing.get("requires_approval", False)

        self._tools[tool_name] = {
            "name": tool_name,
            "func": func,
            "definition": definition,
            "requires_approval": final_approval
        }
        adk_logger.debug(f"Registered tool: {tool_name} (approval={final_approval})")
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
        func = self._tools[name]["func"]
        
        # Handle both sync and async functions if called synchronously
        if inspect.iscoroutinefunction(func):
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If already in a loop, we can't block. This is a fallback risk.
                    adk_logger.warning(f"Sync execution of async tool '{name}' in running loop.")
                    return asyncio.run_coroutine_threadsafe(func(**kwargs), loop).result()
                return asyncio.run(func(**kwargs))
            except RuntimeError:
                return asyncio.run(func(**kwargs))
        
        return func(**kwargs)

    async def aexecute(self, name: str, **kwargs) -> Any:
        """
        Asynchronously executes a registered tool by name.
        """
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in registry.")
            
        adk_logger.info(f"Executing tool (async): {name} with args: {kwargs}")
        func = self._tools[name]["func"]
        
        if inspect.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            # OPTIMIZATION: Offload synchronous blocking tools to a thread
            # so they don't block the async event loop during parallel execution.
            import asyncio
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: func(**kwargs))

# Global tool registry
tool_registry = ToolRegistry()
tool = tool_registry.register
