import asyncio
import os
from litellm_adk import LiteLLMAgent, Session
from litellm_adk.memory import FileMemory

async def run_chat():
    db_path = "session_store.json"
    memory = FileMemory(db_path)
    
    print("\n--- Phase 1: Creating a Premium Session ---")
    # Create session with unique metadata
    session1 = Session(
        id="session_premium_123",
        user_id="alice@example.com",
        metadata={"plan": "enterprise", "region": "US"}
    )
    
    agent = LiteLLMAgent(
        model="oci/xai.grok-3",
        api_key="sk-1234",
        base_url="http://localhost:9000/v1",
        memory=memory,
        system_prompt="You are a helpful assistant who remembers user details.",
        parallel_tool_calls=True,
        sequential_tool_execution=False
    )
    
    # Update local state
    session1.update_state("workflow", "onboarding")
    agent.save_session(session1) # Persist metadata
    
    print(f"Session Initialized: {session1.user_id} (Plan: {session1.metadata['plan']})")
    print(f"Workflow State: {session1.state['workflow']}")

    # Phase 2: Simulating System Restart
    print("\n--- Phase 2: System Restoring Session from Memory ---")
    
    # New agent instance (Pure Service)
    new_agent = LiteLLMAgent(
        model="oci/xai.grok-3",
        api_key="sk-1234",
        base_url="http://localhost:9000/v1",
        memory=memory,
        system_prompt="You are a helpful assistant who remembers user details.",
        parallel_tool_calls=True,
        sequential_tool_execution=False
    )
    
    # Restore session object from memory for inspection
    restored_meta = memory.get_session_metadata("session_premium_123")
    restored_session = Session(**restored_meta)
    
    print(f"Restored User: {restored_session.user_id}")
    print(f"Restored Plan: {restored_session.metadata.get('plan')}")
    print(f"Restored Workflow: {restored_session.state.get('workflow')}")
    
    if restored_session.user_id == "alice@example.com":
        print("✅ SUCCESS: Session properties persisted and restored correctly!")
    else:
        print("❌ FAILED: Session properties lost.")

if __name__ == "__main__":
    asyncio.run(run_chat())
    # Cleanup
    if os.path.exists("session_store.json"):
        os.remove("session_store.json")
