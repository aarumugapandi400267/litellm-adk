import asyncio
from litellm_adk import LiteLLMAgent, tool

async def main():
    # 1. Setup small context limit (e.g. 400 tokens) to force truncation
    @tool
    def get_weather(city: str):
        """Get weather info."""
        return f"The weather in {city} is sunny."

    agent = LiteLLMAgent(
        model="oci/xai.grok-3",
        api_key="sk-1234",
        base_url="http://localhost:9000/v1",
        max_context_tokens=400, # Small limit to trigger context window management
        system_prompt="You are a helpful assistant. Keep your answers brief.",
        tools=[get_weather],
    )

    try:
        session_id = "truncation-test"
        print(f"--- 🧱 Context Truncation Demo (Limit: 400 Tokens) ---")

        # Turn 1: Save a secret
        print("\n[Turn 1] User: My favorite color is 'Indigo'.")
        await agent.ainvoke("My favorite color is 'Indigo'.", session_id=session_id)

        # Turn 2: Noise to fill context
        print("[Turn 2] User: What is the weather in NYC?")
        await agent.ainvoke("What is the weather in NYC?", session_id=session_id)

        # Turn 3: Large text to push Turn 1 out of the window
        noise = "The quick brown fox jumps over the lazy dog. " * 15
        print(f"[Turn 3] User: (Sending 300+ tokens of noise...)")
        await agent.ainvoke(f"Just summarize this briefly: {noise}", session_id=session_id)

        # Turn 4: Final verification
        print("\n[Turn 4] User: What was my favorite color from Turn 1?")
        response = await agent.ainvoke("What was my favorite color from Turn 1?", session_id=session_id)
        print(f"Agent: {response}")
        print("\n(Note: If Turn 1 were truncated, the agent might say it doesn't know.)")

    finally:
        await agent.aclose()

if __name__ == "__main__":
    asyncio.run(main())
