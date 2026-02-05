"""Simple test for debugging the complete function."""

import asyncio
import os

from amplifier_core.message_models import ChatRequest, Message, TextBlock, ToolCallBlock, ToolResultBlock
from amplifier_module_provider_openai_v2 import OpenAIProvider


def test_complete_basic():
    """Basic test to debug the complete function."""
    # Uses OPENAI_API_KEY from environment
    provider = OpenAIProvider(api_key=os.environ.get("OPENAI_API_KEY"))

    messages = [Message(role="user", content="Say hello")]
    request = ChatRequest(messages=messages)

    result = asyncio.run(provider.complete(request))

    print(f"Result: {result}")


def test_complete_with_tool_calls_and_results():
    """Test conversation flow with tool calls and tool results.

    This test simulates a realistic multi-turn conversation where:
    1. User asks to find YAML files
    2. Assistant responds with text and makes a glob tool call
    3. Tool result is returned with file list
    4. User asks a follow-up question

    Based on real Amplifier session data from transcript.jsonl.
    """
    provider = OpenAIProvider(api_key=os.environ.get("OPENAI_API_KEY"))

    messages = [
        # Turn 1: User request
        Message(role="user", content="Can you check which recipe files exist in the project?"),
        # Turn 2: Assistant responds with text and tool call
        Message(
            role="assistant",
            content=[
                TextBlock(type="text", text="Let me search for recipe YAML files in the project."),
                ToolCallBlock(
                    type="tool_call",
                    id="call_abc123xyz",
                    name="glob",
                    input={"pattern": "**/*.yaml", "path": "recipes/"},
                ),
            ],
        ),
        # Turn 3: Tool result
        Message(
            role="function",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    tool_call_id="call_abc123xyz",
                    output='{"files": ["recipes/code-review.yaml", "recipes/document-generation.yaml", "recipes/ecosystem-audit.yaml"], "total_files": 3}',
                )
            ],
        ),
        # Turn 4: User follow-up
        Message(role="user", content="Great, can you summarize what each recipe does based on their names?"),
    ]

    request = ChatRequest(messages=messages)
    result = asyncio.run(provider.complete(request))

    print(f"Result: {result}")
    # Verify we got a response (the model should respond to the follow-up question)
    assert result is not None


if __name__ == "__main__":
    test_complete_basic()
    test_complete_with_tool_calls_and_results()
