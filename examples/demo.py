import asyncio
from litellm_adk.agents import LiteLLMAgent
from litellm_adk.tools import tool

# 1. Define tools using the simplified @tool decorator
@tool
def get_weather(location: str):
    """
    Get the current weather in a given location.
    """
    return f"The weather in {location} is sunny and 25°C."

@tool
def get_stock_price(symbol: str):
    """
    Get the current stock price for a given ticker symbol.
    """
    return f"The stock price of {symbol} is $150.25 (up 1.2%)."

@tool
def calculate_sum(a: int, b: int):
    """
    Adds two numbers together.
    """
    return a + b

@tool
def send_email(to: str, subject: str, body: str):
    """
    Sends an email to a recipient.
    """
    return f"Email sent to {to} with subject: {subject}"

async def main():
    # 2. Initialize Agent with tools as a SIMPLE ARRAY OF FUNCTIONS!
    agent = LiteLLMAgent(
        model="oci/meta.llama-3.1-405b-instruct", 
        system_prompt="You are a helpful assistant. Answer the user's question and call the tools if needed.",
        api_key="sk-1234", 
        base_url="http://localhost:9000/v1",
        tools=[get_weather, get_stock_price, calculate_sum, send_email],
        parallel_tool_calls=False,      
        sequential_tool_execution=True
    )

    # 3. Example of dynamic override and SEQUENTIAL TOOL CALLING
    print("\n--- Tool Calling Invocation ---")
    async for chunk in agent.astream(
        "What is the weather in London & Chennai and also the stock price of AAPL?"
    ):
        print(chunk, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())
