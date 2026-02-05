"""Simple test for debugging the complete function."""

import asyncio
import os

from amplifier_core.message_models import ChatRequest, Message, TextBlock, ToolCallBlock, ToolResultBlock, ToolSpec
from amplifier_module_provider_openai_v2 import OpenAIProvider, RequestTimeoutError


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


def test_complete_background_mode():
    """Test that background mode works for a simple request.

    Background mode offloads the request to OpenAI's async pipeline,
    then polls until the response is ready.
    """
    provider = OpenAIProvider(
        api_key=os.environ.get("OPENAI_API_KEY"),
        config={"background": True},
    )

    messages = [Message(role="user", content="Say hello")]
    request = ChatRequest(messages=messages)

    result = asyncio.run(provider.complete(request))

    print(f"Result: {result}")
    assert result is not None


def test_complete_timeout():
    """Test that a very short timeout raises RequestTimeoutError.

    Sets timeout to 0.5 seconds which is far too short for any real
    API call, so the request should always time out.
    """
    provider = OpenAIProvider(
        api_key=os.environ.get("OPENAI_API_KEY"),
        config={"timeout": 0.5},
    )

    messages = [Message(role="user", content="Say hello")]
    request = ChatRequest(messages=messages)

    try:
        asyncio.run(provider.complete(request))
        assert False, "Expected RequestTimeoutError but request succeeded"
    except RequestTimeoutError:
        print("Correctly raised RequestTimeoutError")


def test_complete_web_search():
    """Test that the built-in web search tool works when enabled.

    Asks a question that requires current information to encourage
    the model to use web search.
    """
    provider = OpenAIProvider(
        api_key=os.environ.get("OPENAI_API_KEY"),
        config={"web_search": True, "default_model": "gpt-5.2", "reasoning": "low"},
    )

    messages = [Message(role="user", content="Can you look up the latest news?")]
    request = ChatRequest(messages=messages)

    result = asyncio.run(provider.complete(request))

    print(f"Result: {result}")
    assert result is not None


def test_complete_multi_turn_with_tool_definitions():
    """Test a real multi-turn tool-use loop by calling complete() twice.

    Exercises the full round-trip:
    1. First complete(): user asks to list files -> model returns tool call(s)
    2. We construct assistant message + fake tool results from the response
    3. Second complete(): pass full history back -> model responds with text

    Tool definitions are passed via ChatRequest.tools as ToolSpec objects.
    """
    provider = OpenAIProvider(api_key=os.environ.get("OPENAI_API_KEY"))

    tools = [
        ToolSpec(
            name="bash",
            description="Execute a shell command and return stdout, stderr, and return code.",
            parameters={
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {"type": "string", "description": "Bash command to execute"},
                },
            },
        ),
        ToolSpec(
            name="read_file",
            description="Read a file from the local filesystem.",
            parameters={
                "type": "object",
                "required": ["file_path"],
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to read"},
                },
            },
        ),
        ToolSpec(
            name="glob",
            description="Find files matching a glob pattern.",
            parameters={
                "type": "object",
                "required": ["pattern"],
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match"},
                    "path": {"type": "string", "description": "Base path to search from"},
                },
            },
        ),
    ]

    # --- First call: user asks something that should trigger tool use ---
    messages = [Message(role="user", content="List the Python files in /tmp/myproject using the glob tool.")]
    request = ChatRequest(messages=messages, tools=tools)

    first_result = asyncio.run(provider.complete(request))
    print(f"First result: {first_result}")
    assert first_result is not None
    assert first_result.tool_calls, "Expected the model to make at least one tool call"

    # --- Build the next turn from the first response ---
    # Assistant message with the content blocks returned by the model
    messages.append(Message(role="assistant", content=first_result.content))

    # Fake tool results for each tool call the model made
    for tc in first_result.tool_calls:
        messages.append(
            Message(
                role="function",
                content=[
                    ToolResultBlock(
                        type="tool_result",
                        tool_call_id=tc.id,
                        output='{"files": ["app.py", "utils.py", "tests/test_app.py"], "total_files": 3}',
                    )
                ],
            )
        )

    # --- Second call: model should summarize the tool results ---
    request2 = ChatRequest(messages=messages, tools=tools)
    second_result = asyncio.run(provider.complete(request2))
    print(f"Second result: {second_result}")
    assert second_result is not None


def test_complete_multi_turn_reasoning_and_tools():
    """Test a multi-turn conversation that requires both reasoning and tool use.

    Presents a debugging scenario where the model must:
    1. Reason about what could cause a test failure
    2. Use tools to investigate (read the source file, then run tests)
    3. Synthesize findings after seeing tool results

    Exercises the full round-trip twice: complete() -> tool results -> complete().
    """
    provider = OpenAIProvider(api_key=os.environ.get("OPENAI_API_KEY"))

    tools = [
        ToolSpec(
            name="read_file",
            description="Read a file from the local filesystem.",
            parameters={
                "type": "object",
                "required": ["file_path"],
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to read"},
                },
            },
        ),
        ToolSpec(
            name="bash",
            description="Execute a shell command and return stdout, stderr, and return code.",
            parameters={
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {"type": "string", "description": "Bash command to execute"},
                },
            },
        ),
    ]

    # --- First call: present a debugging problem that requires reasoning + tool use ---
    messages = [
        Message(
            role="user",
            content=(
                "Our test suite is failing with 'AttributeError: NoneType has no attribute split' "
                "in /tmp/project/src/parser.py. Think step-by-step about what could cause this, "
                "then read the source file to investigate."
            ),
        )
    ]
    request = ChatRequest(messages=messages, tools=tools)

    first_result = asyncio.run(provider.complete(request))
    print(f"First result tool_calls: {first_result.tool_calls}")
    assert first_result is not None
    assert first_result.tool_calls, "Expected the model to call a tool to investigate"

    # --- Build next turn: assistant response + fake tool results ---
    messages.append(Message(role="assistant", content=first_result.content))

    for tc in first_result.tool_calls:
        # Return a fake source file that contains the bug
        output = (
            '{"file_path": "/tmp/project/src/parser.py", "content": "'
            "def parse_header(raw_input):\\n"
            "    # BUG: raw_input can be None when header is missing\\n"
            '    parts = raw_input.split(\\\\\\":\\\\\\\\n\\\\\\")\\n'
            '    return {\\\\\\"key\\\\\\": parts[0], \\\\\\"value\\\\\\": parts[1]}\\n'
            '"}'
        )
        messages.append(
            Message(
                role="function",
                content=[ToolResultBlock(type="tool_result", tool_call_id=tc.id, output=output)],
            )
        )

    # --- Second call: model should reason about the bug and potentially suggest a fix ---
    request2 = ChatRequest(messages=messages, tools=tools)
    second_result = asyncio.run(provider.complete(request2))
    print(f"Second result: {second_result}")
    assert second_result is not None


if __name__ == "__main__":
    test_complete_basic()
    test_complete_with_tool_calls_and_results()
    test_complete_background_mode()
    test_complete_timeout()
    test_complete_web_search()
    test_complete_multi_turn_with_tool_definitions()
    test_complete_multi_turn_reasoning_and_tools()
