import asyncio
import os
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from litellm_adk import LiteLLMAgent, tool, Session
from litellm_adk.core.models import ApprovalStatus
from litellm_adk.core.policy import PolicyEngine

# 1. Define sensitive tools
@tool
def send_funds(recipient: str, amount: float):
    """Sends funds to a recipient."""
    return f"Successfully transferred ${amount} to {recipient}."

@tool
def delete_user_account(user_id: str):
    """Permanently deletes a user account."""
    return f"Account {user_id} has been permanently deleted."

async def main():
    # 2. Configure the Policy Engine
    # We want to catch large payments and ANY account deletions
    policy = PolicyEngine()
    
    # Rule: Payments > $500 require approval
    policy.add_rule(
        "send_funds", 
        condition=lambda args: args.get("amount", 0) > 500,
        description="High-value transfers require oversight."
    )
    
    # Rule: All deletions require approval
    policy.add_rule(
        "delete_user_account",
        description="Destructive actions require mandatory approval."
    )

    agent = LiteLLMAgent(
        model="oci/xai.grok-3", 
        tools=[send_funds, delete_user_account],
        policy_engine=policy
    )

    # 3. Mock the LLM calls so the demo runs without OCI credentials
    agent._aget_completion = AsyncMock()
    
    # Define a helper for fake LLM results using SimpleNamespace
    def mock_resp(content=None, tool_name=None, tool_args=None, tool_id="call_999"):
        tc = None
        if tool_name:
            tc = [SimpleNamespace(
                id=tool_id,
                function=SimpleNamespace(name=tool_name, arguments=json.dumps(tool_args))
            )]
        
        # Structure matches OpenAI response object
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(role="assistant", content=content, tool_calls=tc)
            )]
        )

    session_id = "hitl-demo-session"
    print("--- 🛡️ Advanced HITL Demo: Policy-Based Oversight ---")
    
    try:
        # --- SCENARIO A: Automatic Pass ---
        print("\n[Step 1] Requesting a small payment ($50)...")
        agent._aget_completion.side_effect = [
            mock_resp(tool_name="send_funds", tool_args={"recipient": "Alice", "amount": 50.0}),
            mock_resp(content="Transferred $50 to Alice.")
        ]
        res1 = await agent.ainvoke("Send $50 to Alice.", session_id=session_id)
        print(f"Result: {res1}")

        # --- SCENARIO B: Policy Pause & Modification ---
        print("\n[Step 2] Requesting a LARGE payment ($1200)...")
        pay_id = "call_big_money"
        agent._aget_completion.side_effect = [
            mock_resp(tool_name="send_funds", tool_args={"recipient": "Bob", "amount": 1200.0}, tool_id=pay_id),
            mock_resp(content="Processed modified amount of $450 to Bob.")
        ]
        res2 = await agent.ainvoke("Transfer $1200 to Bob.", session_id=session_id)
        
        if isinstance(res2, dict) and res2.get("status") == "requires_approval":
            print("⏸️ AGENT PAUSED: Tool call requires approval.")
            req = res2["pending_approvals"][0]
            
            # Modify and Submit
            print(f"\n[Reviewer Intervention] Reducing $1200 -> $450.")
            agent.approval_manager.submit_decision(
                id=req["id"],
                status=ApprovalStatus.MODIFIED,
                modified_args={"recipient": "Bob", "amount": 450.0},
                reason="Limit exceeded."
            )
            
            res2_final = await agent.ainvoke("", session_id=session_id)
            print(f"Final Result: {res2_final}")

        # --- SCENARIO C: Mandatory Rejection ---
        print("\n[Step 3] Requesting account deletion...")
        del_id = "call_danger"
        agent._aget_completion.side_effect = [
            mock_resp(tool_name="delete_user_account", tool_args={"user_id": "user_99"}, tool_id=del_id),
            mock_resp(content="I'm sorry, I cannot delete that account per policy.")
        ]
        res3 = await agent.ainvoke("Delete user 'user_99'.", session_id=session_id)
        
        if isinstance(res3, dict) and res3.get("status") == "requires_approval":
            print("⏸️ AGENT PAUSED: Mandatory review triggered.")
            req = res3["pending_approvals"][0]
            
            print("\n[Reviewer Intervention] REJECTING request.")
            agent.approval_manager.submit_decision(
                id=req["id"],
                status=ApprovalStatus.REJECTED,
                reason="Manual direct deletion restricted."
            )
            
            res3_final = await agent.ainvoke("", session_id=session_id)
            print(f"Final Agent Response: {res3_final}")

        print("\n--- 📝 Audit Trail Summary ---")
        if os.path.exists("approvals.json"):
            with open("approvals.json", "r") as f:
                audit_data = json.load(f).get("requests", {})
                for req_id, data in audit_data.items():
                    print(f"ID: {req_id} | Tool: {data['tool_name']} | Status: {data['status']} | Reason: {data['reason']}")

    finally:
        await agent.aclose()
        # Cleanup logs
        if os.path.exists("approvals.json"):
            os.remove("approvals.json")

if __name__ == "__main__":
    asyncio.run(main())
