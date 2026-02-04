import asyncio
import os
from litellm_adk import LiteLLMAgent, Session, tool
from litellm_adk.memory import FileMemory

async def run_chat():
    db_path = "session_store.json"
    memory = FileMemory(db_path)
    
    agent = LiteLLMAgent(
        model="oci/xai.grok-3",
        api_key="sk-1234",
        base_url="http://localhost:9000/v1",
        memory=memory,
        system_prompt="You are a helpful assistant who remembers user details."
    )

    try:
        print("\n--- 🎟️ Phase 1: Creating a Premium Session ---")
        session1 = Session(
            id="session_premium_123",
            user_id="alice@example.com",
            metadata={"plan": "enterprise", "region": "US"}
        )
        
        # Update local state and persist
        session1.update_state("workflow", "onboarding")
        agent.save_session(session1) 
        
        print(f"Session Initialized: {session1.user_id} (Plan: {session1.metadata['plan']})")
        print(f"Workflow State: {session1.state['workflow']}")

        print("\n--- 🔄 Phase 2: Restoring Session from Memory ---")
        # Imagine a new request comes in for the same session ID
        restored_meta = memory.get_session_metadata("session_premium_123")
        if restored_meta:
            restored_session = Session(**restored_meta)
            print(f"Restored User: {restored_session.user_id}")
            print(f"Restored Workflow: {restored_session.state.get('workflow')}")
            print("✅ SUCCESS: Session properties persisted and restored correctly!")
        else:
            print("❌ FAILED: Session not found in memory.")

    finally:
        await agent.aclose()
        # Cleanup demo file
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    asyncio.run(run_chat())
