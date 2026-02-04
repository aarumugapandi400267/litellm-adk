import asyncio
import time
from litellm_adk import LiteLLMAgent, tool

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

async def main():
    agent = LiteLLMAgent(
        model="oci/xai.grok-3", # Mocked or real OCI
        api_key="sk-1234",
        base_url="http://localhost:9000/v1",
        tools=[check_inventory, calculate_shipping]
    )

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
