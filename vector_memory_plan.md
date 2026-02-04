# Vector Memory Implementation Plan

## Objective
Implement Semantic Search (Vector Memory) for the LiteLLM ADK to enable long-term recall of conversations and facts beyond the immediate context window.

## Core Components

1.  **VectorStore Interface (`src/litellm_adk/memory/vector_store.py`)**
    - Abstract base class for vector backends.
    - methods: `add(text, metadata)`, `search(query, k)`.

2.  **Default Backend: ChromaDB (`src/litellm_adk/memory/backends/chroma.py`)**
    - Local, embedded vector database.
    - Persistent storage.

3.  **VectorMemory Class (`src/litellm_adk/memory/vector_memory.py`)**
    - Integrates `VectorStore` into the ADK's memory system.
    - Logic to decide *what* to embed (User messages? Assistant replies? Summaries?).

4.  **Agent Integration**
    - Update `LiteLLMAgent` to check if `memory` supports vector search.
    - **Context Injection**:
        - On `prepare_messages`, query the vector store for relevant past info.
        - Inject this into a dedicated "Relevant Context" block in the System Prompt or as a separate system message.

## Steps

1.  [ ] Create `VectorStore` Protocol/ABC.
2.  [ ] Implement `ChromaVectorStore`.
3.  [ ] Update `LiteLLMAgent` to use `vector_memory.search()` during message preparation.
4.  [ ] Create `examples/vector_memory_demo.py`.

## Dependencies
- `chromadb` (we will need to ask user to install or assume it's there).
- `sentence-transformers` (usually needed for local embeddings, or use OpenAI embeddings via LiteLLM).
    - *Decision*: Use LiteLLM's `embedding()` function so we support ANY provider (OpenAI, Azure, Bedrock) without extra heavy deps if possible.

## Design Detail: Embedding
We will use `litellm.embedding()` to generate vectors. This keeps the ADK unified under LiteLLM's configuration (using the same API key/Base URL if applicable, or a specific embedding model).
