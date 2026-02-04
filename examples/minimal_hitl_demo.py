import asyncio
from litellm_adk import LiteLLMAgent, tool
from litellm_adk.memory import FileMemory
from litellm_adk.core.policy import PolicyEngine

# 1. Define tool (Clean definition, no policy logic inside)
@tool
def send_funds(recipient: str, amount: float):
    """Sends funds."""
    return f"Successfully transferred ${amount} to {recipient}."

async def main():
    # 2. Configure a centralized Policy Engine
    policy = PolicyEngine()
    policy.add_rule(
        "send_funds", 
        condition=lambda args: args.get('amount', 0) > 500,
        description="Transfers > $500 require approval."
    )

    # 3. Initialize Agent with Policy
    agent = LiteLLMAgent(
        model="oci/xai.grok-3", 
        api_key="sk-1234", 
        base_url="http://localhost:9000/v1", 
        tools=[send_funds],
        policy_engine=policy,
        memory=FileMemory("history.json")
    )
    session = "pure-hitl-demo"

    try:
        print("🚀 Requesting $1200 transfer...")
        res = await agent.ainvoke("Send Bob $1200.", session_id=session)

        # 3. Handle the pause if triggered
        if isinstance(res, dict) and res.get("status") == "requires_approval":
            req_id = res["pending_approvals"][0]["id"]
            
            print(f"⏸️  Paused (Policy Exception). Overriding to $450...")
            agent.modify(req_id, {"recipient": "Bob", "amount": 450.0}, reason="Limit")
            
            # 4. Resume - just call again with empty prompt
            final = await agent.ainvoke("", session_id=session)
            print(f"✅ Final Result: {final}")
    finally:
        await agent.aclose()

if __name__ == "__main__":
    asyncio.run(main())