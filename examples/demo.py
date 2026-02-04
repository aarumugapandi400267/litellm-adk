import asyncio
from litellm_adk import LiteLLMAgent, tool

# 1. Define tools using the simplified @tool decorator
@tool
def get_weather(location: str):
    """Get the current weather in a given location."""
    return f"The weather in {location} is sunny and 25°C."

@tool
def get_stock_price(symbol: str):
    """Get the current stock price for a ticker symbol."""
    return f"The stock price of {symbol} is $150.25 (up 1.2%)."

@tool
def calculate_sum(a: int, b: int):
    """Adds two numbers together."""
    return a + b

async def main():
    # 2. Initialize Agent
    agent = LiteLLMAgent(
        model="oci/xai.grok-3", 
        api_key="sk-1234", 
        base_url="http://localhost:9000/v1",
        tools=[get_weather, get_stock_price, calculate_sum]
    )

    try:
        print("\n--- 🚀 LiteLLM ADK Demo ---")
        
        # 3. Simple Invocation
        print("\n[User]: What's the weather in Chennai?")
        response = await agent.ainvoke("What's the weather in Chennai?")
        print(f"[Agent]: {response}")

        # 4. Complex Streaming turn
        print("\n[User]: Calculate 123 + 456 and check AAPL price.")
        print("[Agent]: ", end="", flush=True)
        async for chunk in agent.astream("Calculate 123 + 456 and check AAPL price."):
            print(chunk, end="", flush=True)
        print()

    finally:
        # 5. Proper Cleanup (Crucial for Windows/LiteLLM stability)
        await agent.aclose()

if __name__ == "__main__":
    asyncio.run(main())
