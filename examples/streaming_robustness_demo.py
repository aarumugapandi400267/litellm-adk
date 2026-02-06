import asyncio
import time
from litellm_adk import LiteLLMAgent
from litellm_adk.tools import tool

# 1. Define slow parallel tools to simulate real-world latency
@tool
async def check_inventory(item: str):
    """Checks stock levels (Slow)."""
    await asyncio.sleep(1) # Non-blocking delay
    return f"Stock for {item}: 42 units."

@tool
async def calculate_shipping(zip_code: str):
    """Calculates shipping costs (Slow)."""
    await asyncio.sleep(1.5) # Non-blocking delay
    return f"Shipping to {zip_code}: $12.99."

import os

async def main():
    # Detect if we should use a real model or fallback for demo purposes if not configured
    model = "gemini/gemini-2.5-flash"
    api_key = "sk-1234"
    base_url = "http://localhost:9000/v1"

    agent = LiteLLMAgent(
        model=model,
        api_key=api_key,
        base_url=base_url,
        tools=[check_inventory, calculate_shipping],
        sequential_tool_calls=True
    )

    # --- MOCKING FOR DEMO (Comment out for REAL execution) ---
    # from unittest.mock import MagicMock
    # from types import SimpleNamespace

    # async def mock_asequence_generator(messages):
    #     # Check if we already have tool results in the conversation
    #     last_msg = messages[-1]
    #     if last_msg.get("role") == "tool":
    #         # 2. Second turn: Tools finished, generate final answer
    #         yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Both items checks complete.", tool_calls=None))])
    #         await asyncio.sleep(0.5)
    #         yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=" Verified stock and shipping.", tool_calls=None))])
    #         return

    #     # 1. First turn: User asks question -> Yield "Thinking" then Tool Calls
    #     yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Thinking...", tool_calls=None))])
    #     await asyncio.sleep(0.5)
        
    #     yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
    #         content=None, 
    #         tool_calls=[
    #             SimpleNamespace(index=0, id="call_1", function=SimpleNamespace(name="check_inventory", arguments='{"item": "Titanium Screws"}')),
    #             SimpleNamespace(index=1, id="call_2", function=SimpleNamespace(name="calculate_shipping", arguments='{"zip_code": "90210"}'))
    #         ]
    #     ))])

    # # We mock _aget_completion to return an awaitable that yields the generator
    # # Since 'await agent._aget_completion()' happens, we need an async function that returns the generator
    # async def mock_wrapper(*args, **kwargs):
    #     messages = kwargs.get("messages", [])
    #     return mock_asequence_generator(messages)
    
    # agent._aget_completion = MagicMock(side_effect=mock_wrapper)
    # -------------------------------------------------------------
    # -------------------------------------------------------------

    try:
        print("--- 🚀 STARTING STRUCTURED STREAM ---")
        
        # We use stream_events=True to get real-time tool status
        async for event in agent.astream(
            "Check stock for 'Titanium Screws' and shipping to 90210.", 
            stream_events=True
        ):
            if event["type"] == "content":
                print(event["delta"], end="", flush=True)
            elif event["type"] == "tool_start":
                print(f"\n[🔄 Thinking: Executing {event['name']}...]", end="", flush=True)
            elif event["type"] == "tool_end":
                print(f"\n[✅ Done: {event['name']} returned result]", end="", flush=True)

        print("\n--- ✨ STREAM FINISHED ---")
    finally:
        await agent.aclose()

if __name__ == "__main__":
    asyncio.run(main())
