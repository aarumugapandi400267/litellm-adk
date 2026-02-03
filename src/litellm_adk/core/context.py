from typing import List, Dict, Any, Optional
import litellm
from ..observability.logger import adk_logger

class ContextManager:
    """
    Handles token counting and history truncation/summarization.
    """
    
    @staticmethod
    def count_tokens(messages: List[Dict[str, Any]], model: str) -> int:
        """
        Calculate the number of tokens in a list of messages.
        Uses LiteLLM's token_counter for multi-provider support.
        """
        try:
            # Pass a copy to avoid any potential in-place modifications by token_counter
            return litellm.token_counter(model=model, messages=[m.copy() for m in messages])
        except Exception as e:
            adk_logger.warning(f"Token counting failed for model {model}: {e}. Falling back to estimate.")
            # Rough estimate: 4 chars per token
            return sum(len(str(m.get("content", ""))) for m in messages) // 4

    @staticmethod
    def truncate_history(
        messages: List[Dict[str, Any]], 
        model: str, 
        max_tokens: int,
        reserve_tokens: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Truncate history to fit within max_tokens, always preserving the system prompt
        and the latest message.
        """
        if not messages:
            return []
            
        # 1. Separate System Prompt
        system_prompt = None
        if messages[0].get("role") == "system":
            system_prompt = messages[0]
            other_messages = messages[1:]
        else:
            other_messages = messages

        # 2. Calculate Budget
        # Ensure reserve_tokens doesn't eat the whole budget for small limits
        actual_reserve = min(reserve_tokens, int(max_tokens * 0.2))
        allowed_tokens = max_tokens - actual_reserve
        
        if system_prompt:
            allowed_tokens -= ContextManager.count_tokens([system_prompt], model)

        # 3. Truncate (Keeping the LATEST messages)
        truncated = []
        current_tokens = 0
        
        # Always keep at least the latest message (the current query)
        if other_messages:
            last_msg = other_messages[-1]
            truncated.append(last_msg)
            current_tokens += ContextManager.count_tokens([last_msg], model)
            
            # Work backwards from the second to last message
            for msg in reversed(other_messages[:-1]):
                msg_tokens = ContextManager.count_tokens([msg], model)
                if current_tokens + msg_tokens > allowed_tokens:
                    break
                truncated.insert(0, msg)
                current_tokens += msg_tokens

        # 4. Reconstruct
        result = []
        if system_prompt:
            result.append(system_prompt)
        result.extend(truncated)
        
        return result
