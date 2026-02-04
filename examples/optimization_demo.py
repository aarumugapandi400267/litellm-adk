import asyncio
from litellm_adk import LiteLLMAgent, tool
from litellm_adk.memory import FileMemory

@tool
async def get_stock_price(symbol: str):
    """Fetches the current stock price."""
    await asyncio.sleep(0.5)
    return {"symbol": symbol, "price": 150.0}

@tool
async def get_company_news(symbol: str):
    """Fetches latest news for a company."""
    await asyncio.sleep(0.5)
    return f"News for {symbol}: Strong growth."

async def run_demo():
    # Use FileMemory to show persistence across runs
    memory = FileMemory("optimization_history.json")
    
    agent = LiteLLMAgent(
        model="oci/xai.grok-3",
        api_key="sk-1234",
        base_url="http://localhost:9000/v1",
        tools=[get_stock_price, get_company_news],
        memory=memory,
        parallel_tool_calls=True
    )

    try:
        print("\n--- ⚡ Optimization Demo: Parallel Tool Calls ---")
        print("[User]: Get stock price and news for Apple.")
        
        # Parallel execution will happen if model requests both
        response = await agent.ainvoke("Get stock price and news for Apple.")
        print(f"[Agent]: {response}")

        print("\n--- 🧱 Optimization Demo: Context Management ---")
        session_id = "opt-session"
        await agent.ainvoke("Remember that my secret code is 'X-99'.", session_id=session_id)
        
        # Let's verify it remembers in a second turn
        resp2 = await agent.ainvoke("What is my secret code?", session_id=session_id)
        print(f"[Agent Recall]: {resp2}")

    finally:
        await agent.aclose()

if __name__ == "__main__":
    asyncio.run(run_demo())
