
import asyncio
import logging
import sys
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, ForeignKey
from litellm_adk.integrations.database.agent import DatabaseAgent, NL2SQLAgent
from litellm_adk.integrations.memory.local.file import FileMemory

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nl2sql_demo")

async def main():
    # Setup DB
    db_url = "mongodb://localhost:27017/oci_playground"
    
    
    # 2. Initialize Agent with Config (Demonstrates Adapter Usage)
    db_config = {
        "type": "mongo",
        "url": "mongodb://localhost:27017/oci_playground"
    }
    
    agent = DatabaseAgent(
        db_config=db_config,
        # db_url=db_url, # Legacy way still works if db_config is omitted
        model="oci/xai.grok-3", 
        api_key="sk-1234",
        base_url="http://localhost:9000/v1",
        system_prompt="You are a helpful data analyst.",
        return_direct=True,
        schema_config={'exclude_tables': []},
        data_dictionary={
            "users.country": "ISO 3166-1 alpha-3 code (e.g., USA, GBR)",
            "orders.status": "Order state: 'pending', 'completed', 'failed', 'refunded'",
            "orders.status": "Order state: 'pending', 'completed', 'failed', 'refunded'",
        },
        memory=FileMemory("memory.json"),
        lazy_schema_limit=10 # Example: If > 10 tables, use lazy loading (User has 30+)
    )
    
    # Mocking for demo if no real LLM access provided in environment
    # agent._aget_completion = AsyncMock(...) 
    # But let's assume user will run with valid environment if they have one.
    # Or rely on failover to local model if configured.
    
    try:
        # 3. Test Cases
        print("\n--- Test 1: Simple Aggregation ---")
        prompt1 = "List all sessions?" 
        print(f"User: {prompt1}")
        # Note: In a real run without keys, this will fail or fallback.
        try:
            print("Agent: ", end="", flush=True)
            full_response = ""
            async for chunk in agent.astream(prompt1, stream_events=True):
                if isinstance(chunk, dict):
                    if chunk["type"] == "content":
                        print(chunk["delta"], end="", flush=True)
                        full_response += chunk["delta"]
                    elif chunk["type"] == "tool_start":
                        print(f"\n\n[Tool Call: {chunk['name']}]", end="\nResult: ", flush=True)
                    elif chunk["type"] == "tool_end":
                         # The result is the "Blind Summary" string
                         print(f"{chunk['result']}\n", end="\nAgent: ", flush=True)
                elif isinstance(chunk, str):
                    # Fallback if stream_events logic misses something or returns bare string
                    print(chunk, end="", flush=True)
                    full_response += chunk
            print() # Newline after stream
            if agent.last_query_result:
                print("\n--- 📋 Result Data (UI View) ---")
                # Format as a simple list of records for the demo
                for idx, row in enumerate(agent.last_query_result[:5]): # Show first 5
                     print(f"Record {idx+1}: {row}")
                if len(agent.last_query_result) > 5:
                    print(f"... and {len(agent.last_query_result) - 5} more records.")
        except Exception as e:
            print(f"Agent failed (likely no API key): {e}")
            import traceback
            traceback.print_exc()

    finally:
        await agent.aclose()

if __name__ == "__main__":
    asyncio.run(main())
