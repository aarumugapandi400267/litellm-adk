# LiteLLM ADK (Agent Development Kit)

Highly flexible multiservice Agent Development Kit for building AI agents using LiteLLM.

Built for developers who need to swap models, API keys, and base URLs dynamically while maintaining a robust structure for tool usage, **modular memory persistence**, and observability.

## Features

- **Model Agnostic**: Access 100+ LLMs (OpenAI, Anthropic, OCI Grok-3, Llama, etc.) seamlessly.
- **Easy Tools**: Register Python functions with the `@tool` decorator. No manual JSON schema management.
- **Modular Memory**: Native support for conversation persistence:
    - `InMemoryMemory`: Fast, ephemeral storage.
    - `FileMemory`: Simple JSON-based local persistence.
    - `MongoDBMemory`: Scalable, remote persistence.
- **Parallel & Sequential Execution**: Built-in support for parallel tool calls with robust stream accumulation.
- **Dynamic Configuration**: Global defaults via `.env` or per-agent/per-request overrides.
- **Async & Streaming**: Native support for `ainvoke`, `stream`, and `astream`.

## Installation

```bash
pip install litellm-adk
```

## Quick Start

### Simple Conversational Agent

```python
from litellm_adk.agents import LiteLLMAgent
from litellm_adk.memory import FileMemory

# Setup persistent memory
memory = FileMemory("chat_history.json")

agent = LiteLLMAgent(
    model="gpt-4", 
    memory=memory,
    session_id="user-123"
)

response = agent.invoke("My name is Alice.")
print(agent.invoke("What is my name?")) # Alice
```

### Registering Tools

```python
from litellm_adk.tools import tool

@tool
def get_weather(location: str):
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny."

agent = LiteLLMAgent(tools=[get_weather])
agent.invoke("What is the weather in London?")
```

## Configuration

The ADK uses `pydantic-settings`. Configure via `.env`:

- `ADK_MODEL`: Default model (e.g., `gpt-4o`).
- `ADK_API_KEY`: Default API key.
- `ADK_BASE_URL`: Global base URL override.
- `ADK_LOG_LEVEL`: DEBUG, INFO, etc.

## Documentation
- [Example: Basic Tools](./examples/demo.py)
- [Example: Persistent Memory](./examples/memory_demo.py)

## License

MIT
