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
        # 2. Zero-Config Vector Store
        # By default, this uses local 'all-MiniLM-L6-v2' (dim 384)
        # No extra embedding function code needed!
        vector_store = PostgresVectorStore(
            connection_string=connection_string,
            table_name="adk_memory_demo_v2" # Fresh table
        )
    except ImportError:
        print("⚠️  Skipping demo: 'asyncpg' or 'pgvector' not installed.")
        return
    except Exception as e:
        print(f"⚠️  Initialization failed: {e}")
        return

    # # 3. Seed Memory
    # print("--- 🧠 Vector Memory Demo (Zero-Config) ---")
    # facts = [
    #     "The user's favorite color is Emerald Green.",
    #     "The user lives in a penthouse in New York.",
    #     "The secret project code is 'Project Chimera'."
    # ]
    
    # try:
    #     print("Seeding memory with facts...")
    #     await vector_store.add_texts(facts)
    #     print("✅ Facts embedded and stored.")
    # except Exception as e:
    #     print(f"❌ Failed to seed memory: {e}")

    # 6. Initialize Agent with Vector Store
    # The Agent will automatically use vector_store.search(), which uses the custom provider.
    agent = LiteLLMAgent(
        model="oci/xai.grok-3",
        api_key="sk-1234",
        base_url="http://localhost:9000/v1",
        vector_store=vector_store,
        tools=[get_weather],
        parallel_tool_calls=True,
        fallbacks=["oci/xai.grok-3"],
        system_prompt="You are a helpful assistant.",
    )

    # --- MOCKING FOR DEMO (Pass-through if you have a real server) ---
    # We mock the LLM response to ensure the demo runs without a local server.
    # We will verify that the Vector Context was correctly injected into the prompts.
    # from unittest.mock import AsyncMock
    # from types import SimpleNamespace
    
    # async def mock_completion(*args, **kwargs):
    #     messages = kwargs.get("messages", [])
    #     print(f"\n[MockLLM] Received {len(messages)} messages.")
        
    #     # Check for System Context Injection
    #     system_msgs = [m for m in messages if m.get("role") == "system"]
    #     for m in system_msgs:
    #         if "Emerald Green" in m["content"]:
    #             print(f"[MockLLM] ✅ DETECTED MEMORY CONTEXT:\n{m['content'].strip()}")
        
    #     last_user_msg = messages[-1]["content"]
        
    #     # Simple Logic for Demo Responses
    #     content = "I don't know."
    #     if "favorite color" in last_user_msg:
    #         content = "Based on your memory, your favorite color is Emerald Green and you live in a New York penthouse."
    #     elif "weather" in last_user_msg:
    #          # Just return a helpful response, usually the tool would be called here
    #          # But for this simple vector demo we just want to prove retrieval
    #          content = "The context says you live in New York. Calling weather tool for New York..." 
    #          # In a full tool loop simulation we'd return a tool call here.
    #          # For brevity, we return text confirming retrieval.
        
    #     return SimpleNamespace(
    #         choices=[SimpleNamespace(
    #             message=SimpleNamespace(role="assistant", content=content, tool_calls=None)
    #         )]
    #     )

    # agent._aget_completion = AsyncMock(side_effect=mock_completion)
    # -------------------------------------------------------------

    try:
        # 6. Query about the facts
        print("\n[User]: What is my favorite color and where do I live?")
        response = await agent.ainvoke("Weather in chennai, bangalore and mumbai?")
        print(f"[Agent]: {response}")

    finally:
        await agent.aclose()
        if hasattr(vector_store, "close"):
            await vector_store.close()

if __name__ == "__main__":
    asyncio.run(main())
