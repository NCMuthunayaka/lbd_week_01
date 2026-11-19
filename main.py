from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    assert isinstance(a, (int, float)), "a must be a number"
    assert isinstance(b, (int, float)), "b must be a number"
    return a + b


@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract b from a."""
    assert isinstance(a, (int, float)), "a must be a number"
    assert isinstance(b, (int, float)), "b must be a number"
    return a - b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    assert isinstance(a, (int, float)), "a must be a number"
    assert isinstance(b, (int, float)), "b must be a number"
    return a * b


@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide a by b."""
    assert isinstance(a, (int, float)), "a must be a number"
    assert isinstance(b, (int, float)), "b must be a number"
    assert b != 0, "Cannot divide by zero"
    return a / b


def main():
    model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0, max_retries=1)

    tools = [add_numbers, subtract_numbers, multiply_numbers, divide_numbers]
    agent_executor = create_agent(model=model, tools=tools)

    print("Welcome to the LangChain Agent!", "Type 'exit' to quit.")
    print("You can ask me anything, and I'll do my best to help you.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        result = agent_executor.invoke(
            {"messages": [HumanMessage(content=user_input)]}
        )
        response = result["messages"][-1].content

        # The model may return either plain text or a list of content blocks.
        if isinstance(response, list):
            response = "".join(
                block.get("text", "")
                for block in response
                if isinstance(block, dict)
            )

        print("\nAssistant:", response)


if __name__ == "__main__":
    main()
