import asyncio
from litellm_adk import LiteLLMAgent

async def main():
    # Initialize with a clearly broken model and a valid fallback
    config = {
        "model": "unsupported-model-name",
        "api_key": "sk-1234",
        "base_url": "http://localhost:9000/v1",
        "fallbacks": ["oci/xai.grok-3"]
    }

    print("--- Failover Demo ---")
    print("Primary: unsupported-model-name")
    print("Fallback: oci/xai.grok-3")
    
    try:
        async with LiteLLMAgent(**config) as agent:
            response = await agent.ainvoke("What is 2+2?")
            print(f"Agent Response: {response}")
            print("Result: Pass (Failover successful)")
    except Exception as e:
        print(f"Result: Fail (Error: {e})")

if __name__ == "__main__":
    asyncio.run(main())
