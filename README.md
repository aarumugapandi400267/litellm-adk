# 🤖 LiteLLM ADK (Agent Development Kit)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by LiteLLM](https://img.shields.io/badge/Powered%20by-LiteLLM-blue)](https://github.com/BerriAI/litellm)

A production-grade, highly flexible multiservice framework for building AI agents. Swap models, API keys, and configurations dynamically while maintaining a robust structure for tool usage, complex memory persistence, and human oversight.

---

## 🌟 Key Features

*   **🔌 Model Agnostic**: Native support for 100+ LLMs (OpenAI, Anthropic, OCI Grok-3, Llama) via LiteLLM.
*   **📊 Database Intelligence**: Built-in `DatabaseAgent` (NL2SQL) with adapters for MySQL, PostgreSQL, and MongoDB.
*   **🧠 Advanced Memory**: Pluggable persistence via local JSON, MongoDB, or Vector stores (Chroma, PGVector).
*   **🛡️ Human-in-the-Loop**: Integrated `PolicyEngine` and `ApprovalManager` for safe tool execution with audit trails.
*   **⚡ Modern Async/Streaming**: First-class support for `ainvoke` and event-based streaming (`astream`).
*   **🔍 Observability**: Structured logging and token tracking out of the box.

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install litellm-adk
```

### 2. Basic Agent with Tools
```python
import asyncio
from litellm_adk import LiteLLMAgent, tool, FileMemory

# Define a tool with zero boilerplate
@tool
def get_stock_price(symbol: str):
    """Fetches the current stock price."""
    return {"symbol": symbol, "price": 150.0}

async def main():
    # Initialize with persistent file memory
    agent = LiteLLMAgent(
        model="oci/xai.grok-3",
        tools=[get_stock_price],
        memory=FileMemory("history.json")
    )
    
    response = await agent.ainvoke("What is the price of AAPL?")
    print(f"Agent: {response}")
    await agent.aclose()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📊 High-Performance Database Agents

The ADK specializes in **NL2SQL** via the `DatabaseAgent`. It supports "Blind Summarization" for security—allowing the agent to talk about data without leaking PII into model logs.

```python
from litellm_adk import DatabaseAgent

agent = DatabaseAgent(
    db_url="mysql+pymysql://user:pass@localhost/analytics",
    model="gpt-4o",
    return_direct=True # Privacy: Summary to LLM, full data to your UI
)
```

---

## 📖 Documentation

Dive deeper into the components:

*   [🤖 **Agent Core**](./docs/AGENT.md): Logic, streaming, and failover.
*   [📊 **Database Integration**](./docs/DATABASE_AGENT.md): SQL, MySQL, and MongoDB.
*   [🧠 **Memory & Persistence**](./docs/MEMORY.md): Storing state and history.
*   [🛠️ **Tools & HITL**](./docs/TOOLS.md): Custom tools and Human-in-the-Loop policies.

---

## 🏗️ Project Structure

*   `core/`: Abstract base classes (Agent, Memory, Adapter).
*   `integrations/`: Specific implementations (MySQL, MongoDB, ChromaDB).
*   `session/`: State management for multi-user chat.
*   `observability/`: Logging and telemetry.

---

## 📜 License

MIT © 2026 Aarumugapandi
