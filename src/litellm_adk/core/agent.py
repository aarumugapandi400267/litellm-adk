import uuid
import litellm
import json
import logging
import asyncio
from collections import OrderedDict
from types import SimpleNamespace
from typing import List, Dict, Any, Optional, Union, Callable, Generator, AsyncGenerator

from .base_agent import BaseAgent
from ..observability.logger import adk_logger
from ..config.settings import settings
from ..session import Session
from .tool_registry import tool_registry
from .memory import BaseMemory
from .vector_store import VectorStore
from ..integrations.memory.local.in_memory import InMemoryMemory
from .context import ContextManager
from .approval import ApprovalManager
from .models import ApprovalStatus, ApprovalRequest, AgentResponse
from .policy import PolicyEngine

# Global LiteLLM configuration for resilience
litellm.drop_params = True

class LiteLLMAgent(BaseAgent):
    """
    Multiservice agent supporting dynamic overrides for base_url and api_key.
    """

    async def aclose(self):
        """
        Properly close all global litellm async clients and agent resources.
        This provides a structural resolution to the 'coroutine never awaited' 
        warning often seen on Windows at script exit.
        """
        # 1. Close instance resources
        if getattr(self, "vector_store", None) and hasattr(self.vector_store, "close"):
            try:
                # Check if it's async or sync close
                if asyncio.iscoroutinefunction(self.vector_store.close):
                    await self.vector_store.close()
                else:
                    self.vector_store.close()
                adk_logger.debug("Closed vector_store connection.")
            except Exception as e:
                adk_logger.warning(f"Error closing vector_store: {e}")

        # 2. Close memory resources if applicable
        if getattr(self, "memory", None) and hasattr(self.memory, "close"):
             try:
                if asyncio.iscoroutinefunction(self.memory.close):
                    await self.memory.close()
                else:
                    self.memory.close()
             except Exception:
                 pass

        try:
            # 3. Global LiteLLM Cleanup (Aggressively close any cached clients)
            if hasattr(litellm, "in_memory_llm_clients_cache"):
                cache = litellm.in_memory_llm_clients_cache
                if hasattr(cache, "cache_dict"):
                    for key, client in list(cache.cache_dict.items()):
                        try:
                            if hasattr(client, "aclose"):
                                await client.aclose()
                            elif hasattr(client, "close"):
                                if asyncio.iscoroutinefunction(client.close):
                                    await client.close()
                                else:
                                    client.close()
                        except Exception:
                            pass
                    # Clear the cache so we don't try again or leave refs
                    cache.cache_dict.clear()
            
            # 4. Call official cleanup
            await litellm.close_litellm_async_clients()
            
            # 5. Windows/aiohttp fix: Give time for underlying connections to close
            await asyncio.sleep(0.250)

            # 4. Structural Resolution: Patch litellm's cleanup to be a no-op 
            # This prevents its internal atexit handler from creating un-awaited coroutines
            try:
                import litellm.llms.custom_httpx.async_client_cleanup as cleanup_mod
                
                # We return a completed future so that run_until_complete(close_...) still works
                # but doesn't actually start any new async work or leave coroutines hanging.
                def sync_noop(*args, **kwargs):
                    f = asyncio.Future()
                    f.set_result(None)
                    return f
                
                cleanup_mod.close_litellm_async_clients = sync_noop
            except ImportError:
                pass
            
            adk_logger.debug("Global LiteLLM async clients closed and cache cleared.")
        except Exception as e:
            adk_logger.debug(f"Error during LiteLLM cleanup: {e}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[List[Dict[str, Any]]] = None,
        memory: Optional[BaseMemory] = None,
        max_context_tokens: Optional[int] = None,
        fallbacks: Optional[List[Union[str, Dict[str, Any]]]] = None,
        vector_store: Optional['VectorStore'] = None,
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
                elif isinstance(t, dict):
                     processed_tools.append(t)
            self.tools = processed_tools
        
        # Initialize BaseAgent (Model, Memory, VectorStore)
        resolved_memory = memory or InMemoryMemory()
        super().__init__(
            model=self.model,
            system_prompt=system_prompt,
            memory=resolved_memory,
            vector_store=vector_store
        )
        
        self.max_context_tokens = max_context_tokens
        self.approval_manager = ApprovalManager()
        self.policy_engine = PolicyEngine()
        
        # Setup Fallbacks
        self.fallbacks = fallbacks or []
        
        adk_logger.info(f"Agent initialized with model: {self.model}")
        
        # LRU Cache for Vector Retrieval
        self._context_cache: OrderedDict[str, str] = OrderedDict()
        self._max_cache_size = kwargs.pop("vector_cache_size", 50)
        self.vector_search_threshold = kwargs.pop("vector_search_threshold", None)
        self._max_cache_size = kwargs.pop("vector_cache_size", 50)
        
        # Determine tool execution mode (defaulting to Parallel if not specified)
        parallel_tools = kwargs.pop("parallel_tool_calls", None)
        if parallel_tools is not None:
             self.sequential_tool_execution = not parallel_tools
        else:
             self.sequential_tool_execution = kwargs.pop("sequential_tool_execution", settings.sequential_execution)

        self.extra_kwargs = kwargs
        
        if "parallel_tool_calls" not in self.extra_kwargs:
            self.extra_kwargs["parallel_tool_calls"] = True
            
        # Memory Persistence
        self.memory = memory or InMemoryMemory()
        self.max_context_tokens = max_context_tokens
        
        # Process fallbacks: ensure they are standardized
        self.fallbacks = []
        if fallbacks:
            for f in fallbacks:
                config = {"model": f} if isinstance(f, str) else f.copy()
                
                # Apply model prefixing to fallbacks if using custom base_url (and fallback doesn't override it)
                target_base_url = config.get("base_url", self.base_url)
                if target_base_url and not config["model"].startswith("openai/"):
                    config["model"] = f"openai/{config['model']}"
                    
                self.fallbacks.append(config)
        
        adk_logger.debug(f"Initialized LiteLLMAgent as a service for model={self.model}")

    def save_session(self, session: Union[str, Session]):
        """Persist session metadata and state to memory."""
        actual_id = session.id if isinstance(session, Session) else session
        
        # If it's a Session object, we dump the full metadata
        if isinstance(session, Session):
            self.memory.save_session_metadata(actual_id, session.model_dump())
        # If it's just an ID, there's nothing to dump from the service layer

    # --- HITL Convenience Methods ---
    def approve(self, request_id: str, reviewer: str = "human", reason: Optional[str] = None):
        """Approve a pending tool call."""
        self.approval_manager.submit_decision(request_id, ApprovalStatus.APPROVED, reviewer, reason)

    def reject(self, request_id: str, reviewer: str = "human", reason: Optional[str] = None):
        """Reject a pending tool call."""
        self.approval_manager.submit_decision(request_id, ApprovalStatus.REJECTED, reviewer, reason)

    def modify(self, request_id: str, modified_args: Dict[str, Any], reviewer: str = "human", reason: Optional[str] = None):
        """Provide modified arguments and approve the tool call."""
        self.approval_manager.submit_decision(request_id, ApprovalStatus.MODIFIED, reviewer, reason, modified_args)

    async def _retrieve_context(self, prompt: str) -> Optional[str]:
        """Async semantic search for relevant context with LRU Caching."""
        if not self.vector_store or not prompt:
            return None
        
        # 1. Check Cache
        if prompt in self._context_cache:
            adk_logger.debug(f"Cache hit for vector context: {prompt[:30]}...")
            self._context_cache.move_to_end(prompt)
            return self._context_cache[prompt]
            
        try:
            results = await self.vector_store.search(prompt, k=3, score_threshold=self.vector_search_threshold)
            if not results:
                return None
            
            adk_logger.info(f"Retrieved {len(results)} chunks for context:")
            for i, r in enumerate(results):
                content = r.get('text', '')
                preview = content[:200] + "..." if len(content) > 200 else content
                adk_logger.info(f"Chunk {i+1}: {preview}")
            
            context_block = "\n".join([f"- {r['text']}" for r in results])
            result_str = f"Relevant Context from Memory:\n{context_block}"
            
            # 2. Update Cache
            self._context_cache[prompt] = result_str
            self._context_cache.move_to_end(prompt)
            if len(self._context_cache) > self._max_cache_size:
                self._context_cache.popitem(last=False)
                
            return result_str
        except Exception as e:
            adk_logger.warning(f"Vector retrieval failed: {e}")
            return None


    def _prepare_messages(self, prompt: str, actual_session_id: str) -> List[Dict[str, str]]:
        # 2. Fetch/Initialize History from Memory
        history = self.memory.get_messages(actual_session_id)
        is_new_session = not history
        
        if is_new_session:
            history = [{"role": "system", "content": self.system_prompt}]
            # Don't persist system prompt until first real turn to keep DB clean
            
        messages = history.copy()
        if prompt:
            # 2a. Vector Retrieval (Semantic Search)
            if self.vector_store:
                try:
                    # Search logic inside _prepare_messages might be blocking in sync invoke, 
                    # but wait, vector_store.search is async.
                    # LiteLLMAgent methods are invoke(sync) and ainvoke(async).
                    # 'invoke' cannot await. So VectorStore mostly supports async agents.
                    # We will only support it for async invoke for now or use asyncio.run/loop for sync (risky).
                    # Actually, for simplicity in 'prepare_messages' which is sync:
                    # We might need to skip distinct vector search here or make prepare_messages async?
                    # But prepare_messages is called by sync invoke. 
                    # Let's put a TODO/Warning for sync usage or assume async usage is primary.
                    pass
                except Exception as e:
                     adk_logger.warning(f"Vector search failed: {e}")

            user_msg = {"role": "user", "content": prompt}
            messages.append(user_msg)
            
            # 3. Persist turn start
            # Ensure messages are sanitized and tokenized before first persistence
            current_user_msg = self._sanitize_message(user_msg)
            current_user_msg["token_count"] = ContextManager.count_tokens([current_user_msg], self.model)
            
            if is_new_session:
                system_msg = self._sanitize_message(messages[0])
                system_msg["token_count"] = ContextManager.count_tokens([system_msg], self.model)
                self.memory.add_messages(actual_session_id, [system_msg, current_user_msg])
            else:
                self.memory.add_message(actual_session_id, current_user_msg)
        
        # 4. Context Management (Truncation)
        if self.max_context_tokens:
            messages = ContextManager.truncate_history(
                messages, 
                self.model, 
                self.max_context_tokens
            )
            
        return messages

    def _update_history(self, new_messages: List[Dict[str, Any]], actual_session_id: str):
        """Persist new messages to memory with token counts."""
        if new_messages:
            sanitized = []
            for m in new_messages:
                s = self._sanitize_message(m)
                # Compute token count if not already present (optimization for future turns)
                if "token_count" not in s:
                    s["token_count"] = ContextManager.count_tokens([s], self.model)
                sanitized.append(s)
                
            self.memory.add_messages(actual_session_id, sanitized)

    def _sanitize_message(self, message: Any) -> Dict[str, Any]:
        """
        Convert LiteLLM message objects to strictly compliant dictionaries.
        Ensures compatibility with strict providers like OCI.
        """
        # If it's already a dict, extract only what we need to avoid 'extra key' errors
        role = getattr(message, "role", "assistant") if not isinstance(message, dict) else message.get("role", "assistant")
        content = getattr(message, "content", "") if not isinstance(message, dict) else message.get("content", "")
        
        # OCI/OpenAI standard: content cannot be None for assistant/user/system
        if content is None:
            content = ""
            
        msg_dict = {
            "role": role,
            "content": content
        }
        
        # Handle Tool Calls (Assistant Message)
        tool_calls = getattr(message, "tool_calls", None) if not isinstance(message, dict) else message.get("tool_calls")
        if tool_calls:
            msg_dict["tool_calls"] = [self._sanitize_tool_call(tc) for tc in tool_calls]
            
        # Handle Tool Result (Tool Role)
        if role == "tool":
            msg_dict["tool_call_id"] = getattr(message, "tool_call_id", None) if not isinstance(message, dict) else message.get("tool_call_id")
            # Name is optional but good practice
            name = getattr(message, "name", None) if not isinstance(message, dict) else message.get("name")
            if name:
                msg_dict["name"] = name
                
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

    def _should_require_approval(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Centralized check for tool approval requirements."""
        # 1. Check registry flag (can be bool OR predicate)
        from ..tools.registry import tool_registry
        registry_meta = tool_registry._tools.get(tool_name, {})
        req = registry_meta.get("requires_approval")
        
        if callable(req):
            if req(arguments): return True
        elif req is True:
            return True
            
        # 2. Check Policy Engine rules
        if self.policy_engine.should_require_approval(tool_name, arguments):
            return True
            
        return False

    async def _aexecute_tool(self, tool_call) -> Dict[str, Any]:
        """Helper to execute a tool call asynchronously and return formatted result."""
        function_name = self._get_tc_val(tool_call, "function", "name")
        raw_args = self._get_tc_val(tool_call, "function", "arguments") or "{}"
        t_id = self._get_tc_val(tool_call, "id")
        
        arguments = self._parse_arguments(raw_args)
        return await self._aexecute_tool_with_args(function_name, t_id, arguments)

    async def _aexecute_tool_with_args(self, tool_name: str, tool_call_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Core async tool execution logic."""
        result = await tool_registry.aexecute(tool_name, **arguments)
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": str(result)
        }

    def _execute_tool(self, tool_call) -> str:
        """Helper to execute a tool call synchronously."""
        function_name = self._get_tc_val(tool_call, "function", "name")
        raw_args = self._get_tc_val(tool_call, "function", "arguments") or "{}"
        arguments = self._parse_arguments(raw_args)
        return self._execute_tool_with_args(function_name, arguments)

    def _execute_tool_with_args(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Core sync tool execution logic."""
        return tool_registry.execute(tool_name, **arguments)

    def _parse_arguments(self, args: Any) -> Dict[str, Any]:
        """Robustly parses tool arguments."""
        if isinstance(args, dict):
            return args
        try:
            return json.loads(args or "{}")
        except json.JSONDecodeError:
            adk_logger.warning(f"Failed to parse tool arguments: {args}")
            return {}

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

    def _get_completion(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs):
        """Execute a completion call with automatic failover support."""
        configs = [{"model": self.model, "api_key": self.api_key, "base_url": self.base_url}] + self.fallbacks
        
        last_err = None
        for config in configs:
            try:
                model = config.get("model")
                api_key = config.get("api_key", self.api_key)
                base_url = config.get("base_url", self.base_url)
                
                return litellm.completion(
                    model=model,
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    tools=tools,
                    **{**self.extra_kwargs, **kwargs}
                )
            except Exception as e:
                # Only failover on specific recoverable errors
                recoverable = (
                    "rate_limit" in str(e).lower() or 
                    "timeout" in str(e).lower() or 
                    "service_unavailable" in str(e).lower() or
                    "internal_server_error" in str(e).lower() or
                    isinstance(e, (litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.APIError, litellm.BadRequestError))
                )
                
                if recoverable and config != configs[-1]:
                    adk_logger.warning(f"Model {model} failed with recoverable error: {e}. Switching to fallback...")
                    last_err = e
                    continue
                raise e
        raise last_err

    async def _aget_completion(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs):
        """Execute an async completion call with automatic failover support."""
        configs = [{"model": self.model, "api_key": self.api_key, "base_url": self.base_url}] + self.fallbacks
        
        last_err = None
        for config in configs:
            try:
                model = config.get("model")
                api_key = config.get("api_key", self.api_key)
                base_url = config.get("base_url", self.base_url)
                
                return await litellm.acompletion(
                    model=model,
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    tools=tools,
                    **{**self.extra_kwargs, **kwargs}
                )
            except Exception as e:
                recoverable = (
                    "rate_limit" in str(e).lower() or 
                    "timeout" in str(e).lower() or 
                    "service_unavailable" in str(e).lower() or
                    "internal_server_error" in str(e).lower() or
                    isinstance(e, (litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.APIError, litellm.BadRequestError))
                )
                
                if recoverable and config != configs[-1]:
                    adk_logger.warning(f"Model {model} failed with recoverable error: {e}. Switching to fallback...")
                    last_err = e
                    continue
                raise e
        raise last_err

    def invoke(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, session_id: Optional[Union[str, Session]] = None, **kwargs) -> Union['AgentResponse', Dict[str, Any]]:
        """
        Execute a synchronous completion with automatic tool calling.
        """
        actual_session_id = session_id.id if isinstance(session_id, Session) else (session_id or str(uuid.uuid4()))
        messages = self._prepare_messages(prompt, actual_session_id=actual_session_id)
        tools = tools or self.tools
        new_turns = [] # Track only what's new in this specific call
        accumulated_content = []
        executed_tool_calls = []
        
        adk_logger.info(f"Invoking completion for model: {self.model}")
        
        while True:
            # RESUME LOGIC
            last_msg = messages[-1]
            if not prompt and len(new_turns) == 0 and last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
                adk_logger.info("Resuming from pending tool calls...")
                message = last_msg
                tool_calls_from_llm = last_msg.get("tool_calls", [])
            else:
                response = self._get_completion(messages=messages, tools=tools, **kwargs)
                message = response.choices[0].message
                tool_calls_from_llm = getattr(message, "tool_calls", [])

            if tool_calls_from_llm:
                pending_requests = []
                for tc in tool_calls_from_llm:
                    t_name = self._get_tc_val(tc, "function", "name")
                    t_id = self._get_tc_val(tc, "id")
                    t_args = self._parse_arguments(self._get_tc_val(tc, "function", "arguments"))
                    
                    # Check if this tool already has a decision in the ApprovalManager
                    request = self.approval_manager.get_request(t_id)
                    
                    if not request:
                        # NEW TOOL CALL: Check if it requires approval
                        if self._should_require_approval(t_name, t_args):
                            request = self.approval_manager.create_request(t_id, actual_session_id, t_name, t_args)
                    
                    if request and request.status == ApprovalStatus.PENDING:
                        pending_requests.append(request)

                if pending_requests:
                    # Atomic Pause: if any tool in batch is pending, pause the whole turn
                    if last_msg != self._sanitize_message(message):
                        sanitized_msg = self._sanitize_message(message)
                        self.memory.add_message(actual_session_id, sanitized_msg)
                    
                    return {
                        "status": "requires_approval",
                        "pending_approvals": [r.model_dump(mode='json') for r in pending_requests],
                        "session_id": actual_session_id
                    }

                # If we get here, all tools are either safe or have a final decision (APPROVED/REJECTED/MODIFIED)
                tool_calls_to_process = [tool_calls_from_llm[0]] if self._should_handle_sequentially() else tool_calls_from_llm

                if self._should_handle_sequentially():
                    if isinstance(message, dict):
                        message["tool_calls"] = tool_calls_to_process
                    else:
                        message.tool_calls = tool_calls_to_process

                if last_msg != self._sanitize_message(message):
                    sanitized_msg = self._sanitize_message(message)
                    messages.append(sanitized_msg)
                    new_turns.append(sanitized_msg)
                    if sanitized_msg.get("content"):
                         accumulated_content.append(sanitized_msg["content"].strip())
                
                for tool_call in tool_calls_to_process:
                    executed_tool_calls.append(self._sanitize_tool_call(tool_call))
                    
                    t_id = self._get_tc_val(tool_call, "id")
                    request = self.approval_manager.get_request(t_id)
                    if request and request.status == ApprovalStatus.REJECTED:
                        result = f"Error: Tool call REJECTED by human reviewer. Reason: {request.reason or 'Not specified.'}"
                    elif request and request.status == ApprovalStatus.MODIFIED:
                        current_raw_args = self._get_tc_val(tool_call, "function", "arguments") or "{}"
                        current_t_args = self._parse_arguments(current_raw_args)
                        final_args = self.approval_manager.get_effective_args(t_id, default_args=current_t_args)
                        raw_result = self._execute_tool_with_args(self._get_tc_val(tool_call, "function", "name"), final_args)
                        # Prefix with context so LLM knows a human changed it
                        result = f"[HUMAN OVERRIDE: Parameters were modified by a reviewer for safety/compliance] {raw_result}"
                    else:
                        current_raw_args = self._get_tc_val(tool_call, "function", "arguments") or "{}"
                        current_t_args = self._parse_arguments(current_raw_args)
                        final_args = self.approval_manager.get_effective_args(t_id, default_args=current_t_args)
                        result = self._execute_tool_with_args(self._get_tc_val(tool_call, "function", "name"), final_args)
                    
                    tool_response = {
                        "role": "tool",
                        "tool_call_id": t_id,
                        "name": self._get_tc_val(tool_call, "function", "name"),
                        "content": str(result)
                    }
                    messages.append(tool_response)
                    new_turns.append(tool_response)
                
                continue
            
            final_msg = self._sanitize_message(message)
            messages.append(final_msg)
            new_turns.append(final_msg)
            if final_msg.get("content"):
                 accumulated_content.append(final_msg["content"].strip())
            
            self._update_history(new_turns, actual_session_id=actual_session_id)
            
            return AgentResponse(
                content=final_msg.get("content") or "",
                accumulated_content="\n".join(accumulated_content),
                tool_calls=executed_tool_calls,
                session_id=actual_session_id
            )

    async def ainvoke(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, session_id: Optional[Union[str, Session]] = None, **kwargs) -> Union['AgentResponse', Dict[str, Any]]:
        """
        Execute an asynchronous completion with automatic tool calling.
        """
        actual_session_id = session_id.id if isinstance(session_id, Session) else (session_id or str(uuid.uuid4()))
        messages = self._prepare_messages(prompt, actual_session_id=actual_session_id)
        
        # Inject Vector Context (Async)
        if prompt and self.vector_store:
            context_str = await self._retrieve_context(prompt)
            if context_str:
                messages.insert(1, {"role": "system", "content": context_str})

        tools = tools or self.tools
        new_turns = []
        accumulated_content = []
        executed_tool_calls = []
        
        adk_logger.info(f"Invoking async completion for model: {self.model}")
        
        while True:
            # RESUME LOGIC
            last_msg = messages[-1]
            if not prompt and len(new_turns) == 0 and last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
                adk_logger.info("Resuming from pending tool calls (async)...")
                message = last_msg
                tool_calls_from_llm = last_msg.get("tool_calls", [])
            else:
                response = await self._aget_completion(messages=messages, tools=tools, **kwargs)
                message = response.choices[0].message
                tool_calls_from_llm = getattr(message, "tool_calls", [])
            
            if tool_calls_from_llm:
                pending_requests = []
                for tc in tool_calls_from_llm:
                    t_name = self._get_tc_val(tc, "function", "name")
                    t_id = self._get_tc_val(tc, "id")
                    t_args = self._parse_arguments(self._get_tc_val(tc, "function", "arguments"))
                    
                    request = self.approval_manager.get_request(t_id)
                    if not request:
                        if self._should_require_approval(t_name, t_args):
                            request = self.approval_manager.create_request(t_id, actual_session_id, t_name, t_args)
                    
                    if request and request.status == ApprovalStatus.PENDING:
                        pending_requests.append(request)

                if pending_requests:
                    if last_msg != self._sanitize_message(message):
                        sanitized_msg = self._sanitize_message(message)
                        self.memory.add_message(actual_session_id, sanitized_msg)
                    return {
                        "status": "requires_approval",
                        "pending_approvals": [r.model_dump(mode='json') for r in pending_requests],
                        "session_id": actual_session_id
                    }

                tool_calls_to_process = [tool_calls_from_llm[0]] if self._should_handle_sequentially() else tool_calls_from_llm

                if self._should_handle_sequentially():
                    if isinstance(message, dict):
                        message["tool_calls"] = tool_calls_to_process
                    else:
                        message.tool_calls = tool_calls_to_process

                if last_msg != self._sanitize_message(message):
                    sanitized_msg = self._sanitize_message(message)
                    messages.append(sanitized_msg)
                    new_turns.append(sanitized_msg)
                    if sanitized_msg.get("content"):
                         accumulated_content.append(sanitized_msg["content"].strip())
                
                if self._should_handle_sequentially():
                    for tool_call in tool_calls_to_process:
                        executed_tool_calls.append(self._sanitize_tool_call(tool_call))
                        t_id = self._get_tc_val(tool_call, "id")
                        request = self.approval_manager.get_request(t_id)
                        
                        if request and request.status == ApprovalStatus.REJECTED:
                            content = f"Error: Tool call REJECTED by human reviewer. Reason: {request.reason or 'Not specified.'}"
                            res = {"role": "tool", "tool_call_id": t_id, "name": self._get_tc_val(tool_call, "function", "name"), "content": content}
                        elif request and request.status == ApprovalStatus.MODIFIED:
                            current_raw_args = self._get_tc_val(tool_call, "function", "arguments") or "{}"
                            current_t_args = self._parse_arguments(current_raw_args)
                            final_args = self.approval_manager.get_effective_args(t_id, default_args=current_t_args)
                            raw_res = await self._aexecute_tool_with_args(self._get_tc_val(tool_call, "function", "name"), t_id, final_args)
                            # Prefix with context
                            raw_res["content"] = f"[HUMAN OVERRIDE: Parameters were modified by a reviewer for safety/compliance] {raw_res['content']}"
                            res = raw_res
                        else:
                            current_raw_args = self._get_tc_val(tool_call, "function", "arguments") or "{}"
                            current_t_args = self._parse_arguments(current_raw_args)
                            final_args = self.approval_manager.get_effective_args(t_id, default_args=current_t_args)
                            res = await self._aexecute_tool_with_args(self._get_tc_val(tool_call, "function", "name"), t_id, final_args)
                        messages.append(res)
                        new_turns.append(res)
                else:
                    # Parallel Execution
                    import asyncio
                    results = []
                    for tc in tool_calls_to_process:
                        executed_tool_calls.append(self._sanitize_tool_call(tc))
                        t_id = self._get_tc_val(tc, "id")
                        request = self.approval_manager.get_request(t_id)
                        
                        if request and request.status == ApprovalStatus.REJECTED:
                             results.append({"role": "tool", "tool_call_id": t_id, "name": self._get_tc_val(tc, "function", "name"), "content": f"Error: Tool call REJECTED by human reviewer."})
                        elif request and request.status == ApprovalStatus.MODIFIED:
                             current_raw_args = self._get_tc_val(tc, "function", "arguments") or "{}"
                             current_t_args = self._parse_arguments(current_raw_args)
                             final_args = self.approval_manager.get_effective_args(t_id, default_args=current_t_args)
                             
                             async def execute_and_prefix(t_name, t_call_id, args):
                                 r = await self._aexecute_tool_with_args(t_name, t_call_id, args)
                                 r["content"] = f"[HUMAN OVERRIDE: Parameters were modified by a reviewer] {r['content']}"
                                 return r
                                 
                             results.append(execute_and_prefix(self._get_tc_val(tc, "function", "name"), t_id, final_args))
                        else:
                             current_raw_args = self._get_tc_val(tc, "function", "arguments") or "{}"
                             current_t_args = self._parse_arguments(current_raw_args)
                             final_args = self.approval_manager.get_effective_args(t_id, default_args=current_t_args)
                             results.append(self._aexecute_tool_with_args(self._get_tc_val(tc, "function", "name"), t_id, final_args))
                    
                    # Gather coroutines
                    actual_results = await asyncio.gather(*[r for r in results if not isinstance(r, dict)])
                    
                    idx = 0
                    for i in range(len(results)):
                        if isinstance(results[i], dict):
                            messages.append(results[i])
                            new_turns.append(results[i])
                        else:
                            messages.append(actual_results[idx])
                            new_turns.append(actual_results[idx])
                            idx += 1
                continue
            
            final_msg = self._sanitize_message(message)
            messages.append(final_msg)
            new_turns.append(final_msg)
            if final_msg.get("content"):
                 accumulated_content.append(final_msg["content"].strip())
            
            self._update_history(new_turns, actual_session_id=actual_session_id)
            
            return AgentResponse(
                content=final_msg.get("content") or "",
                accumulated_content="\n".join(accumulated_content),
                tool_calls=executed_tool_calls,
                session_id=actual_session_id
            )
    def stream(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, session_id: Optional[Union[str, Session]] = None, stream_events: bool = False, **kwargs) -> Generator[Union[str, Dict[str, Any]], None, None]:
        """
        Execute a streaming completion with automatic tool calling.
        If stream_events=True, yields structured dictionaries instead of strings.
        """
        actual_session_id = session_id.id if isinstance(session_id, Session) else (session_id or str(uuid.uuid4()))
        messages = self._prepare_messages(prompt, actual_session_id=actual_session_id)
        tools = tools or self.tools
        
        new_turns = []
        
        while True:
            response = self._get_completion(messages=messages, tools=tools, stream=True, **kwargs)
            
            # Accumulate tool call parts
            full_content = ""
            tool_calls_by_index = {} # map of index -> list of SimpleNamespace
            notified_tools = set()

            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content
                    yield {"type": "content", "delta": delta.content} if stream_events else delta.content
                
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
                        
                        # Yield "Thinking" event as soon as name is known
                        current_tc = tool_calls_by_index[idx][-1]
                        if current_tc.function.name and idx not in notified_tools:
                            if stream_events:
                                yield {"type": "tool_start", "name": current_tc.function.name, "index": idx}
                            notified_tools.add(idx)

            # Build final flat tool calls list
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
  
                assistant_msg = self._sanitize_message({"role": "assistant", "tool_calls": tool_calls, "content": full_content})
                messages.append(assistant_msg)
                new_turns.append(assistant_msg)
                
                # Execute sequentially (sync stream is always sequential execution in practice for simplicity)
                for tool_call in tool_calls:
                    t_name = tool_call["function"]["name"]
                    result = self._execute_tool(tool_call)
                    
                    if stream_events:
                         yield {"type": "tool_end", "name": t_name, "result": str(result)}
                         
                    tool_resp = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": t_name,
                        "content": str(result)
                    }
                    messages.append(tool_resp)
                    new_turns.append(tool_resp)
                
                continue
            
            final_msg = self._sanitize_message({"role": "assistant", "content": full_content})
            messages.append(final_msg)
            new_turns.append(final_msg)
            self._update_history(new_turns, actual_session_id=actual_session_id)
            return

    async def astream(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, session_id: Optional[Union[str, Session]] = None, stream_events: bool = False, **kwargs) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """
        Execute an asynchronous streaming completion with automatic tool calling.
        If stream_events=True, yields structured dictionaries instead of strings.
        """
        actual_session_id = session_id.id if isinstance(session_id, Session) else (session_id or str(uuid.uuid4()))
        messages = self._prepare_messages(prompt, actual_session_id=actual_session_id)
        
        # Inject Vector Context (Async)
        if prompt and self.vector_store:
            context_str = await self._retrieve_context(prompt)
            if context_str:
                messages.insert(1, {"role": "system", "content": context_str})
                
        tools = tools or self.tools
        new_turns = []
        
        while True:
            response = await self._aget_completion(messages=messages, tools=tools, stream=True, **kwargs)
            
            full_content = ""
            tool_calls_by_index = {}
            notified_tools = set()

            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content
                    yield {"type": "content", "delta": delta.content} if stream_events else delta.content
                
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
                        
                        # Yield "Thinking" event as soon as name is known
                        current_tc = tool_calls_by_index[idx][-1]
                        if current_tc.function.name and idx not in notified_tools:
                            if stream_events:
                                yield {"type": "tool_start", "name": current_tc.function.name, "index": idx}
                            notified_tools.add(idx)

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
  
                assistant_msg = self._sanitize_message({"role": "assistant", "tool_calls": tool_calls, "content": full_content})
                messages.append(assistant_msg)
                new_turns.append(assistant_msg)
                
                if self._should_handle_sequentially():
                    for tool_call in tool_calls:
                        t_name = tool_call["function"]["name"]
                        result = await self._aexecute_tool(tool_call)
                        
                        if stream_events:
                             yield {"type": "tool_end", "name": t_name, "result": str(result["content"])}

                        messages.append(result)
                        new_turns.append(result)
                else:
                    # Parallel Execution - YIELD AS THEY FINISH
                    pending = [self._aexecute_tool(tc) for tc in tool_calls]
                    results_to_append = []
                    
                    for coro in asyncio.as_completed(pending):
                        res = await coro
                        t_name = res["name"]
                        if stream_events:
                            yield {"type": "tool_end", "name": t_name, "result": str(res["content"])}
                        results_to_append.append(res)
                    
                    # Sort results back to original order for consistent history (optional but cleaner)
                    # We can use tool_call_id or just append as they come
                    for res in results_to_append:
                        messages.append(res)
                        new_turns.append(res)

                # Continue turn after tool results
                continue
            
            final_msg = self._sanitize_message({"role": "assistant", "content": full_content})
            messages.append(final_msg)
            new_turns.append(final_msg)
            self._update_history(new_turns, actual_session_id=actual_session_id)
            return
