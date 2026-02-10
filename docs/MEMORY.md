# 🧠 Memory & Persistence

ADK provides a flexible memory system to persist conversation history and session metadata across different storage backends.

## 📁 Storage Backends

### 1. Local Memory
*   `InMemoryMemory`: Fastest, stateless (clears on restart).
*   `FileMemory`: Persists history to a local JSON file. Great for local development.

### 2. Remote Memory
*   `MongoDBMemory`: Stores history in a MongoDB collection. Ideal for multi-node deployments.

### 3. Vector Memory
Vector stores allow for semantic search across long-term history.
*   `ChromaVectorStore`: Integrated vector database.
*   `PostgresVectorStore`: Uses `pgvector` for enterprise-grade search.
*   `MongoVectorStore`: Uses Atlas Vector Search.

## 🎟️ Sessions
Sessions wrap user-specific data and state.

```python
from litellm_adk import Session

session = Session(
    id="user_123",
    metadata={"plan": "premium"},
    state={"step": "onboarding"}
)
```

## 🚀 Setup

```python
from litellm_adk import LiteLLMAgent, FileMemory

memory = FileMemory("database.json")
agent = LiteLLMAgent(model="openai/gpt-4o", memory=memory)

# Multi-user isolation happens automatically via session_id
await agent.ainvoke("Remember my name is Alice", session_id="alice_99")
```
