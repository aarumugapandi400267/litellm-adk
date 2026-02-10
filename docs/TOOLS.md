# 🛠️ Tools & Human-in-the-Loop (HITL)

ADK makes tool execution safe and manageable via a centralized registry and an optional approval system.

## ✍️ Defining Tools
Use the `@tool` decorator to automatically generate JSON schemas for any Python function.

```python
from litellm_adk import tool

@tool
def send_email(recipient: str, body: str):
    """Sends an email message."""
    # Logic here
    return "Email sent."
```

## 🛡️ Policy & Approvals
For sensitive actions (e.g., payments, deletions), you can trigger a mandatory human review.

### 1. Configure a Policy
```python
from litellm_adk.core.policy import PolicyEngine

policy = PolicyEngine()
policy.add_rule(
    "send_email",
    condition=lambda args: "@competitor.com" in args.get("recipient", ""),
    description="Prevent emails to competitors."
)
```

### 2. Approval Workflow
When a policy is triggered, `ainvoke` returns a state: `requires_approval`.

```python
res = await agent.ainvoke("Email ceo@competitor.com", session_id="s1")

if res.get("status") == "requires_approval":
    request_id = res["pending_approvals"][0]["id"]
    # Decision: approve(), reject(), or modify()
    agent.reject(request_id, reason="Security policy")
```

## 📋 Audit Trail
All approval decisions are persisted (by default to `approvals.json`) creating a permanent audit trail of AI-human collaboration.
