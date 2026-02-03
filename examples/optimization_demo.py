import asyncio
from litellm_adk import LiteLLMAgent

async def get_stock_price(symbol: str):
    """Fetches the current stock price (simulated with 1s delay)."""
    await asyncio.sleep(1.0)
    return {"symbol": symbol, "price": 150.0}

async def get_company_news(symbol: str):
    """Fetches latest news for a company (simulated with 1s delay)."""
    await asyncio.sleep(1.0)
    return f"News for {symbol}: Strong growth."

async def run_demo():
    agent = LiteLLMAgent(
        model="oci/xai.grok-3",
        api_key="sk-1234",
        base_url="http://localhost:9000/v1",
        tools=[get_stock_price, get_company_news],
        max_context_tokens=1000
    )

    print("\n--- Testing Parallel Tool Calls ---")
    response = await agent.ainvoke("Get stock price and news for Apple and Google.")
    print(f"Agent: {response}")

    print("\n--- Testing Context Truncation ---")
    session_id = "minimal-demo"
    for i in range(3):
        await agent.ainvoke(f"The secret key for level {i} is {i*100}", session_id=session_id)
    
    response = await agent.ainvoke("What was the secret key for level 0?", session_id=session_id)
    print(f"Agent: {response}")

if __name__ == "__main__":
    asyncio.run(run_demo())
