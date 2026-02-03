import asyncio
from litellm_adk import LiteLLMAgent, Session
from litellm_adk.tools import tool
from litellm_adk.memory import InMemoryMemory

# 1. Define a tool
@tool
def get_weather(location: str):
    """Get the current weather in a given location."""
    return f"The weather in {location} is sunny and 25°C."

async def main():
    # 2. Setup Memory (Stateless Service Pattern)
    memory = InMemoryMemory()
    
    # 3. Initialize Agent as a "Service"
    # Note: No session_id provided here. This agent can serve ANY session.
    agent = LiteLLMAgent(
        model="oci/xai.grok-3",
        api_key="sk-1234",
        base_url="http://localhost:9000/v1",
        memory=memory,
        system_prompt="You are a helpful travel assistant.",
        parallel_tool_calls=True,
        sequential_tool_execution=False
    )

    print("--- Stateless Agent Service Initialized ---")

    # 4. User A (Alice) starts a chat
    alice_session = Session(user_id="alice_99", metadata={"loyalty": "gold"})
    print(f"\n[Alice Session: {alice_session.id}]")
    resp_a = await agent.ainvoke("Hi, I'm Alice. I'm planning a trip to Paris.", session_id=alice_session)
    print(f"Agent to Alice: {resp_a}")

    # 5. User B (Bob) starts a chat through the SAME agent instance
    bob_session = Session(user_id="bob_88")
    print(f"\n[Bob Session: {bob_session.id}]")
    resp_b = await agent.ainvoke("Hello, what is the weather in London?", session_id=bob_session)
    print(f"Agent to Bob: {resp_b}")

    # 6. User A returns - Agent remembers her!
    print(f"\n[Alice Returns: {alice_session.id}]")
    resp_a2 = await agent.ainvoke("Where was I planning to go?", session_id=alice_session)
    print(f"Agent to Alice: {resp_a2}")

    # 7. Bob checks recall
    print(f"\n[Bob Returns: {bob_session.id}]")
    resp_b2 = await agent.ainvoke("Where did I ask about?", session_id=bob_session)
    print(f"Agent to Bob: {resp_b2}")

if __name__ == "__main__":
    asyncio.run(main())
