import litellm
import json
from typing import List, Dict, Any, Optional, Union, Generator, AsyncGenerator
from .base import BaseAgent
from ..observability.logger import adk_logger
from ..config.settings import settings
from ..tools.registry import tool_registry
from ..memory.base import BaseMemory
from ..memory.in_memory import InMemoryMemory
import uuid

# Global LiteLLM configuration for resilience
litellm.drop_params = True

class LiteLLMAgent(BaseAgent):
    """
    Multiservice agent supporting dynamic overrides for base_url and api_key.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[List[Dict[str, Any]]] = None,
        memory: Optional[BaseMemory] = None,
        session_id: Optional[str] = None,
        **kwargs
    ):
        self.model = model or settings.model
        self.api_key = api_key or settings.api_key
        self.base_url = base_url or settings.base_url
        
        # Automatically prepend 'openai/' if a base_url is used to force proxy/OpenAI-compatible routing
        if self.base_url and not self.model.startswith("openai/"):
            adk_logger.debug(f"Custom base_url detected. Prepending 'openai/' to model {self.model}")
            self.model = f"openai/{self.model}"
            
        self.system_prompt = system_prompt
        
        # Smart Tool Resolution
        if tools is None:
            # Default to all registered tools if none provided
            self.tools = tool_registry.get_tool_definitions()
        else:
            # Process provided list (can be definitions OR functions)
            processed_tools = []
            for t in tools:
                if callable(t):
                    # It's a function, register it (if not already) and get definition
                    processed_tools.append(tool_registry._register_function(t))
                else:
                    # It's already a definition dict
                    processed_tools.append(t)
            self.tools = processed_tools

        self.extra_kwargs = kwargs
        
        # Ensure model-specific parameters        # Default parallel_tool_calls if not explicitly provided
        if "parallel_tool_calls" not in self.extra_kwargs:
            self.extra_kwargs["parallel_tool_calls"] = False
            
        self.sequential_tool_execution = kwargs.get("sequential_tool_execution", settings.sequential_execution)
        
        # Memory Persistence
        self.memory = memory or InMemoryMemory()
        self.session_id = session_id or str(uuid.uuid4())
        self.history = self.memory.get_messages(self.session_id)
        
        if not self.history:
            self.history = [{"role": "system", "content": self.system_prompt}]
            self.memory.add_message(self.session_id, self.history[0])
            
        adk_logger.debug(f"Initialized LiteLLMAgent with session_id={self.session_id}, model={self.model}")

    def _prepare_messages(self, prompt: str) -> List[Dict[str, str]]:
        # Refresh from memory in case it was modified elsewhere
        self.history = self.memory.get_messages(self.session_id)
        
        messages = self.history.copy()
        user_msg = {"role": "user", "content": prompt}
        messages.append(user_msg)
        
        # Persist the user message immediately
        self.memory.add_message(self.session_id, user_msg)
        self.history.append(user_msg)
        
        return messages

    def _update_history(self, final_messages: List[Dict[str, Any]]):
        """Sync internal history and memory with the final message state."""
        # Find which messages were added since we prepared (the user message was already added)
        # We assume messages order is preserved
        start_idx = len(self.history)
        new_messages = [self._sanitize_message(m) for m in final_messages[start_idx:]]
        
        if new_messages:
            self.memory.add_messages(self.session_id, new_messages)
            self.history.extend(new_messages)

    def _sanitize_message(self, message: Any) -> Dict[str, Any]:
        """Convert LiteLLM message objects to plain dictionaries for serialization."""
        if isinstance(message, dict):
            # Still need to sanitize tool_calls inside if they are objects
            if "tool_calls" in message and message["tool_calls"]:
                message["tool_calls"] = [self._sanitize_tool_call(tc) for tc in message["tool_calls"]]
            return message
        
        # Manually extract common fields to ensure clean JSON
        msg_dict = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None)
        }
        
        if hasattr(message, "name") and message.name:
            msg_dict["name"] = message.name
            
        if hasattr(message, "tool_calls") and message.tool_calls:
            msg_dict["tool_calls"] = [self._sanitize_tool_call(tc) for tc in message.tool_calls]
            
        if hasattr(message, "tool_call_id") and message.tool_call_id:
            msg_dict["tool_call_id"] = message.tool_call_id
            
        return msg_dict

    def _sanitize_tool_call(self, tc: Any) -> Dict[str, Any]:
        """Convert a tool call object to a standard dictionary."""
        if isinstance(tc, dict):
            return tc
            
        tc_dict = {
            "id": getattr(tc, "id", None),
            "type": getattr(tc, "type", "function"),
            "function": {
                "name": None,
                "arguments": ""
            }
        }
        
        func = getattr(tc, "function", None)
        if func:
            tc_dict["function"]["name"] = getattr(func, "name", None)
            tc_dict["function"]["arguments"] = getattr(func, "arguments", "")
            
        return tc_dict

    def _should_handle_sequentially(self) -> bool:
        """Determines if we should process tool calls one by one."""
        return self.sequential_tool_execution

    async def _aexecute_tool(self, tool_call) -> Any:
        # Same as _execute_tool but for async if needed in future
        return self._execute_tool(tool_call)

    def _get_tc_val(self, tool_call, attr, subattr=None):
        """Helper to get value from either object or dict tool call."""
        if isinstance(tool_call, dict):
            val = tool_call.get(attr)
            if val and subattr:
                return val.get(subattr)
            return val
        else:
            val = getattr(tool_call, attr, None)
            if val and subattr:
                return getattr(val, subattr, None)
            return val

    def _execute_tool(self, tool_call) -> Any:
        """Helper to execute a tool call and handle JSON parsing."""
        function_name = self._get_tc_val(tool_call, "function", "name")
        raw_args = self._get_tc_val(tool_call, "function", "arguments") or "{}"
        
        try:
            if isinstance(raw_args, dict):
                arguments = raw_args
            else:
                # Try standard parsing
                arguments = json.loads(raw_args)
        except json.JSONDecodeError:
            # RECOVERY: Handle concatenated JSON objects like {"a":1}{"b":2}
            if isinstance(raw_args, str) and "}{" in raw_args:
                try:
                    # Take only the first valid JSON object
                    decoder = json.JSONDecoder()
                    arguments, _ = decoder.raw_decode(raw_args)
                except Exception:
                    adk_logger.error(f"Failed to recover tool arguments: {raw_args}")
                    arguments = {}
            else:
                adk_logger.warning(f"Failed to parse tool arguments for {function_name}: {raw_args}")
                arguments = {}
        
        return tool_registry.execute(function_name, **arguments)

    def invoke(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
        """
        Execute a synchronous completion with automatic tool calling.
        """
        messages = self._prepare_messages(prompt)
        tools = tools or self.tools
        
        adk_logger.info(f"Invoking completion for model: {self.model}")
        
        while True:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
                tools=tools,
                **{**self.extra_kwargs, **kwargs}
            )
            
            message = response.choices[0].message
            
            # Check if the model wants to call tools
            if hasattr(message, "tool_calls") and message.tool_calls:
                # If sequential is enabled, we only process the FIRST tool call
                tool_calls_to_process = [message.tool_calls[0]] if self._should_handle_sequentially() else message.tool_calls
                
                # We update the original message to only include the calls we are handling 
                # (to keep history clean for strict models)
                if self._should_handle_sequentially():
                    message.tool_calls = tool_calls_to_process

                sanitized_msg = self._sanitize_message(message)
                messages.append(sanitized_msg)
                
                for tool_call in tool_calls_to_process:
                    result = self._execute_tool(tool_call)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": str(result)
                    })
                
                continue
            
            messages.append(self._sanitize_message(message))
            self._update_history(messages)
            return message.content

    async def ainvoke(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
        """
        Execute an asynchronous completion with automatic tool calling.
        """
        messages = self._prepare_messages(prompt)
        tools = tools or self.tools
        
        adk_logger.info(f"Invoking async completion for model: {self.model}")
        
        while True:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
                tools=tools,
                **{**self.extra_kwargs, **kwargs}
            )
            
            message = response.choices[0].message
            
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls_to_process = [message.tool_calls[0]] if self._should_handle_sequentially() else message.tool_calls
                
                if self._should_handle_sequentially():
                    message.tool_calls = tool_calls_to_process

                sanitized_msg = self._sanitize_message(message)
                messages.append(sanitized_msg)
                
                for tool_call in tool_calls_to_process:
                    result = self._execute_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": str(result)
                    })
                continue
            
            messages.append(self._sanitize_message(message))
            self._update_history(messages)
            return message.content

    def stream(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Generator[str, None, None]:
        """
        Execute a streaming completion with automatic tool calling.
        """
        messages = self._prepare_messages(prompt)
        tools = tools or self.tools
        
        while True:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
                stream=True,
                tools=tools,
                **{**self.extra_kwargs, **kwargs}
            )
            
            # Accumulate tool call parts
            full_content = ""
            tool_calls_by_index = {} # map of index -> list of SimpleNamespace
            
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content
                    yield delta.content
                
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_by_index:
                            tool_calls_by_index[idx] = []
                        
                        last_tc = tool_calls_by_index[idx][-1] if tool_calls_by_index[idx] else None
                        
                        # Decide if we need a new tool call object for this index
                        start_new = False
                        if last_tc is None:
                            start_new = True
                        else:
                            # Start new if name is present and last one already has a name
                            if tc_delta.function and tc_delta.function.name and last_tc.function.name:
                                start_new = True
                            # Start new if ID is present and last one already has a different ID
                            elif tc_delta.id and last_tc.id and tc_delta.id != last_tc.id:
                                start_new = True
                        
                        if start_new:
                            from types import SimpleNamespace
                            new_tc = SimpleNamespace(
                                id=tc_delta.id,
                                function=SimpleNamespace(
                                    name=tc_delta.function.name if tc_delta.function else None,
                                    arguments=tc_delta.function.arguments if tc_delta.function else ""
                                )
                            )
                            tool_calls_by_index[idx].append(new_tc)
                        else:
                            # Update existing tool call
                            if tc_delta.id:
                                last_tc.id = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    last_tc.function.name = (last_tc.function.name or "") + tc_delta.function.name
                                if tc_delta.function.arguments:
                                    if last_tc.function.arguments is None:
                                        last_tc.function.arguments = ""
                                    last_tc.function.arguments += tc_delta.function.arguments

            # Build final flattened tool calls list (as dicts for history)
            tool_calls = []
            for idx in sorted(tool_calls_by_index.keys()):
                for tc_obj in tool_calls_by_index[idx]:
                    if tc_obj.function.name:
                        tool_calls.append({
                            "id": tc_obj.id,
                            "type": "function",
                            "function": {
                                "name": tc_obj.function.name,
                                "arguments": tc_obj.function.arguments
                            }
                        })

            if tool_calls:
                # If sequential, only keep the first tool call
                if self._should_handle_sequentially():
                    tool_calls = [tool_calls[0]]

                # Add the assistant's composite tool call message to history
                assistant_msg = {"role": "assistant", "tool_calls": tool_calls, "content": full_content or None}
                messages.append(assistant_msg)
                
                for tool_call in tool_calls:
                    result = self._execute_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "content": str(result)
                    })
                
                # Loop back to continue the conversation with tool results
                continue
            
            # No tool calls, store final content and finish
            messages.append({"role": "assistant", "content": full_content})
            self._update_history(messages)
            return

    async def astream(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> AsyncGenerator[str, None]:
        """
        Execute an asynchronous streaming completion with automatic tool calling.
        """
        messages = self._prepare_messages(prompt)
        tools = tools or self.tools
        
        while True:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
                stream=True,
                tools=tools,
                **{**self.extra_kwargs, **kwargs}
            )
            
            full_content = ""
            tool_calls_by_index = {}
            
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content
                    yield delta.content
                
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_by_index:
                            tool_calls_by_index[idx] = []
                        
                        last_tc = tool_calls_by_index[idx][-1] if tool_calls_by_index[idx] else None
                        start_new = False
                        if last_tc is None:
                            start_new = True
                        else:
                            if tc_delta.function and tc_delta.function.name and last_tc.function.name:
                                start_new = True
                            elif tc_delta.id and last_tc.id and tc_delta.id != last_tc.id:
                                start_new = True
                        
                        if start_new:
                            from types import SimpleNamespace
                            new_tc = SimpleNamespace(
                                id=tc_delta.id,
                                function=SimpleNamespace(
                                    name=tc_delta.function.name if tc_delta.function else None,
                                    arguments=tc_delta.function.arguments if tc_delta.function else ""
                                )
                            )
                            tool_calls_by_index[idx].append(new_tc)
                        else:
                            if tc_delta.id:
                                last_tc.id = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    last_tc.function.name = (last_tc.function.name or "") + tc_delta.function.name
                                if tc_delta.function.arguments:
                                    if last_tc.function.arguments is None:
                                        last_tc.function.arguments = ""
                                    last_tc.function.arguments += tc_delta.function.arguments

            tool_calls = []
            for idx in sorted(tool_calls_by_index.keys()):
                for tc_obj in tool_calls_by_index[idx]:
                    if tc_obj.function.name:
                        tool_calls.append({
                            "id": tc_obj.id,
                            "type": "function",
                            "function": {
                                "name": tc_obj.function.name,
                                "arguments": tc_obj.function.arguments
                            }
                        })

            if tool_calls:
                if self._should_handle_sequentially():
                    tool_calls = [tool_calls[0]]

                assistant_msg = {"role": "assistant", "tool_calls": tool_calls, "content": full_content or None}
                messages.append(assistant_msg)
                
                for tool_call in tool_calls:
                    result = self._execute_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "content": str(result)
                    })
                continue
            
            messages.append({"role": "assistant", "content": full_content})
            self._update_history(messages)
            return
