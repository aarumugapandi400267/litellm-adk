# 🤖 LiteLLMAgent

`LiteLLMAgent` is the core class of the ADK, providing a high-level interface for LLM interactions with integrated tool use, memory, and failover support.

## 🚀 Initialization

```python
from litellm_adk import LiteLLMAgent

agent = LiteLLMAgent(
    model="oci/xai.grok-3",
    system_prompt="You are a helpful assistant.",
    tools=[my_tool],
    memory=my_memory,
    parallel_tool_calls=True
)
```

## 🛠️ Key Features

### 1. Unified Interface
Supports both synchronous `invoke()` and asynchronous `ainvoke()`.

### 2. Multi-Service Failover
Configured via `fallbacks`. If the primary model fails, the agent automatically tries the next one in the list.

### 3. Context Management
Automatically handles token counting and history truncation to fit within the model's context window.

### 4. Streaming Support
Supports event-based streaming via `astream()`, allowing you to track tool calls and partial responses in real-time.

## 📝 Methods

| Method | Description |
|--------|-------------|
| `invoke(prompt, session_id)` | Synchronous chat execution. |
| `ainvoke(prompt, session_id)` | Asynchronous chat execution (Recommended). |
| `astream(prompt, session_id)` | Returns an async generator for event-based streaming. |
| `save_session(session)` | Persists session metadata to memory. |
| `aclose()` | Properly closes async resources (use in `finally` blocks). |
