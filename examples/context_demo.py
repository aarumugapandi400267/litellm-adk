import asyncio
from litellm_adk import LiteLLMAgent
from litellm_adk.session import Session
from litellm_adk.memory import InMemoryMemory
from litellm_adk.tools import tool

async def main():
    # 1. Setup small context limit (e.g. 200 tokens) to force truncation

    @tool
    def get_weather(city):
        return f"The weather in {city} is sunny."

    agent = LiteLLMAgent(
        model="oci/xai.grok-3",
        api_key="sk-1234",
        base_url="http://localhost:9000/v1",
        max_context_tokens=1000, # Small limit for Grok/OAI counting
        system_prompt="You are a poet who remembers colors.",
        tools=[get_weather],
    )

    session_id = "color-chat"
    
    print(f"--- Context Truncation Demo ---")

    # Turn 1: Add some context
    # print("\n[Turn 1]")
    # await agent.ainvoke("My favorite color is Blue.", session_id=session_id)
    # print("User: wHAT IS THE WEATHER IN NEW YORK?")

    # # Turn 2: Add more context
    # print("\n[Turn 2]")
    # await agent.ainvoke("My second favorite color is Red.", session_id=session_id)
    # print("User: My second favorite color is Red.")

    # # Turn 3: Add more context to push the limit
    # print("\n[Turn 3]")
    # long_text = "The quick brown fox jumps over the lazy dog. "
    # await agent.ainvoke(f"Repeat this text: {long_text}", session_id=session_id)
    # print("User: (Sent a long text to push the limit)")

    # # Turn 4: Check if the agent remembers the FIRST color
    # # If truncation worked, 'Blue' should be gone, but 'Red' (more recent) might stay.
    # print("\n[Turn 4]")
    response = await agent.ainvoke("Weather in Chennai.", session_id=session_id)
    print(f"Agent: {response}")

if __name__ == "__main__":
    asyncio.run(main())
