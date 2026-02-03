import asyncio
from litellm_adk.agents import LiteLLMAgent
from litellm_adk.tools import tool
from litellm_adk.memory import MongoDBMemory, InMemoryMemory, FileMemory

# 1. Define a simple tool
@tool
def get_user_name(user_id: str):
    """Get the user's registered name."""
    return "Alice" if user_id == "123" else "Unknown User"

async def main():
    # 2. Setup persistent file memory
    memory = MongoDBMemory(
        connection_string="mongodb://localhost:27017/",
        database_name="my_agent_db",
        collection_name="conversations"
    )
    
    # 3. Initialize Agent with a specific session ID
    session_id = "alice-session-1"
    
    agent = LiteLLMAgent(
        # Uses default model from .env if not specified
        model="oci/xai.grok-3",
        api_key="sk-1234",
        base_url="http://localhost:9000/v1",
        memory=memory,
        session_id=session_id,
        system_prompt="You are a helpful assistant who remembers user details.",
        parallel_tool_calls=True,
        sequential_tool_execution=False
    )

    print(f"--- Session: {session_id} ---")
    
    # Turn 1
    print("\n[Turn 1]")
    resp1 = await agent.ainvoke("Hi, my user ID is 123. What is my name?")
    print(f"Agent: {resp1}")
    
    # Turn 2 (Context check)
    print("\n[Turn 2]")
    resp2 = await agent.ainvoke("Can you remember my name and suggest a fruit I might like?")
    print(f"Agent: {resp2}")

    print("\n--- Memory file 'chat_history.json' updated! ---")

if __name__ == "__main__":
    asyncio.run(main())
