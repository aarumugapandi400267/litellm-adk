# 📖 LiteLLM ADK Documentation

Welcome to the LiteLLM Agent Development Kit (ADK) documentation. This framework is designed to build production-ready AI agents with native support for database interaction, complex memory, and human oversight.

## 🗺️ Documentation Map

*   [🤖 **LiteLLMAgent**](./AGENT.md): Core agent logic, streaming, and failover.
*   [📊 **DatabaseAgent**](./DATABASE_AGENT.md): Natural language to SQL/Mongo (NL2SQL) and data adapters.
*   [🧠 **Memory & Persistence**](./MEMORY.md): Storing history in JSON, MongoDB, or Vector stores.
*   [🛠️ **Tools & HITL**](./TOOLS.md): Defining tools and setting up human-in-the-loop policies.

## 🏗️ Architecture Overview

The ADK is divided into `core` and `integrations`:
*   `core/`: Contains the abstract base classes and logic for Agents, Memory, and Tools.
*   `integrations/`: Contains specific implementations (e.g., MySQL, MongoDB, ChromaDB).

## 🚀 Quick Start (Production Pattern)

```python
import asyncio
from litellm_adk import DatabaseAgent, FileMemory

async def main():
    agent = DatabaseAgent(
        db_url="mysql+pymysql://root:pass@localhost/db",
        model="oci/xai.grok-3",
        memory=FileMemory("chat_log.json"),
        return_direct=True # Security: Data stays in DB, summary to LLM
    )
    
    response = await agent.ainvoke("How many users signed up last month?")
    print(f"Agent: {response}")
    await agent.aclose()

if __name__ == "__main__":
    asyncio.run(main())
```
