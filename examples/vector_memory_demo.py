import asyncio
import os
from litellm_adk import LiteLLMAgent, tool
from litellm_adk.memory.backends.postgres import PostgresVectorStore

# 1. Define tools
@tool
def get_weather(city: str):
    """Get the current weather for a city."""
    return f"The weather in {city} is mild and sunny."

async def main():
    # Use standard PG connection string
    connection_string = os.getenv("PG_URI", "postgresql://postgres:postgres@localhost:5433/postgres")
    
    print(f"Connecting to Postgres at {connection_string}...")
    
    try:
        # 2. Initialize Vector Store (Postgres + pgvector)
        vector_store = PostgresVectorStore(
            connection_string=connection_string,
            table_name="adk_memory_demo",
            embedding_model="azure/text-embedding-3-small" # Using mock/azure for demo
        )
    except ImportError:
        print("⚠️  Skipping demo: 'asyncpg' or 'pgvector' not installed.")
        return
    except Exception as e:
        print(f"⚠️  Initialization failed: {e}")
        return

    # 3. Seed Memory
    print("--- 🧠 Vector Memory Demo (Postgres) ---")
    facts = [
        "The user's favorite color is Emerald Green.",
        "The user lives in a penthouse in New York.",
        "The secret project code is 'Project Chimera'."
    ]
    
    try:
        print("Seeding memory with facts...")
        await vector_store.add_texts(facts)
        print("✅ Facts embedded and stored.")
    except Exception as e:
        print(f"❌ Failed to seed memory (DB Connection or Embedding Error): {e}")
        # We continue to show Agent init even if DB fails, though retrieval won't work.

    # 4. Initialize Agent with Vector Store
    agent = LiteLLMAgent(
        model="oci/xai.grok-3",
        api_key="sk-1234",
        base_url="http://localhost:9000/v1",
        vector_store=vector_store,
        tools=[get_weather]
    )

    try:
        # 5. Query about the facts
        print("\n[User]: What is my favorite color and where do I live?")
        response = await agent.ainvoke("What is my favorite color and where do I live?")
        print(f"[Agent]: {response}")

    finally:
        await agent.aclose()
        if hasattr(vector_store, "close"):
            await vector_store.close()

if __name__ == "__main__":
    asyncio.run(main())
