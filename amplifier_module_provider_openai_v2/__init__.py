"""
OpenAI provider module for Amplifier.
Integrates with OpenAI's Responses API.
"""

__all__ = ["mount", "OpenAIProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import asyncio
import json
import logging
import os
import time
from typing import Any, Literal

import openai
from openai import AsyncOpenAI
from openai._types import omit
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseIncludable,
    ResponseInputItemParam,
    ResponseTextConfigParam,
    response_create_params,
)
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.responses.response_function_tool_call_param import (
    ResponseFunctionToolCallParam,
)
from openai.types.responses.response_input_image_param import ResponseInputImageParam
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputContentParam,
)
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_reasoning_item_param import (
    ResponseReasoningItemParam,
)
from openai.types.responses.function_tool_param import FunctionToolParam
from openai.types.responses.tool_param import ToolParam
from openai.types.shared_params.reasoning import Reasoning
from pydantic import BaseModel

from amplifier_core import ConfigField, ModelInfo, ModuleCoordinator, ProviderInfo
from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    Usage,
    ToolSpec,
    ContentBlockUnion,
)

from ._constants import (
    DEFAULT_DEBUG_TRUNCATE_LENGTH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEEP_RESEARCH_MODELS,
)

logger = logging.getLogger(__name__)


class OpenAIRequest(BaseModel):
    input: list[ResponseInputItemParam]
    model: str
    include: list[ResponseIncludable] | None = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    parallel_tool_calls: bool | None = None
    reasoning: Reasoning | None = None
    temperature: float | None = None
    text: ResponseTextConfigParam | None = None
    tool_choice: response_create_params.ToolChoice | None = None
    tools: list[ToolParam] | None = None  # Must be list, not Iterable, for proper serialization
    truncation: Literal["auto", "disabled"] | None = None
    background: bool | None = None


class ContextLimitExceededError(Exception):
    """Raised when the input context exceeds the provider's limit."""


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the OpenAI provider."""
    config = config or {}

    # Get API key from config or environment
    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        logger.warning("No API key found for OpenAI provider")
        return None

    provider = OpenAIProvider(api_key=api_key, config=config, coordinator=coordinator)
    await coordinator.mount("providers", provider, name="openai")
    logger.info("Mounted OpenAIProvider (Responses API)")

    # Return cleanup function
    async def cleanup():
        if hasattr(provider.client, "close"):
            await provider.client.close()

    return cleanup


class OpenAIProvider:
    """OpenAI Responses API integration."""

    name = "openai"
    api_label = "OpenAI"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
        client: AsyncOpenAI | None = None,
    ):
        """Initialize OpenAI provider with Responses API client.

        The SDK client is created lazily on first use, allowing get_info()
        to work without valid credentials.
        """
        self._api_key = api_key
        self._client: AsyncOpenAI | None = client  # Lazy init if None
        self.config = config or {}
        self.coordinator = coordinator

        # Configuration with sensible defaults (from _constants.py - single source of truth)
        self.base_url = self.config.get("base_url", None)  # Optional custom endpoint (None = OpenAI default)
        self.default_model = self.config.get("default_model", DEFAULT_MODEL)
        self.max_tokens = self.config.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.reasoning: Any | None = self.config.get(
            "reasoning", DEFAULT_REASONING_EFFORT
        )  # None = not sent (minimal|low|medium|high)
        self.debug = self.config.get("debug", False)  # Enable full request/response logging
        self.raw_debug = self.config.get("raw_debug", False)  # Enable ultra-verbose raw API I/O logging
        self.debug_truncate_length = self.config.get(
            "debug_truncate_length", DEFAULT_DEBUG_TRUNCATE_LENGTH
        )  # Max chars before truncation in debug logs
        self.filtered = self.config.get("filtered", True)  # Filter to curated model list by default
        self.background = False

        # Provider priority for selection (lower = higher priority)
        self.priority = self.config.get("priority", 100)

        # Track tool call IDs that have been repaired with synthetic results.
        # This prevents infinite loops when the same missing tool results are
        # detected repeatedly across LLM iterations (since synthetic results
        # are injected into request.messages but not persisted to message store).
        self._repaired_tool_ids: set[str] = set()

    @property
    def client(self) -> AsyncOpenAI:
        """Lazily initialize the OpenAI client on first access."""
        if self._client is None:
            if self._api_key is None:
                raise ValueError("api_key or client must be provided for API calls")
            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self.base_url)
        return self._client

    def get_info(self) -> ProviderInfo:
        """Get provider metadata."""
        return ProviderInfo(
            id="openai",
            display_name="OpenAI",
            credential_env_vars=["OPENAI_API_KEY"],
            capabilities=["streaming", "tools", "reasoning"],
            defaults={
                "model": "gpt-5.2",
                "max_tokens": 64_000,
                "timeout": 300.0,
                "context_window": 400_000,
                "max_output_tokens": 64_000,
            },
            config_fields=[
                ConfigField(
                    id="api_key",
                    display_name="API Key",
                    field_type="secret",
                    prompt="Enter your OpenAI API key",
                    env_var="OPENAI_API_KEY",
                ),
                ConfigField(
                    id="base_url",
                    display_name="API Base URL",
                    field_type="text",
                    prompt="API base URL",
                    env_var="OPENAI_BASE_URL",
                    required=False,
                    default="https://api.openai.com/v1",
                ),
                ConfigField(
                    id="reasoning_effort",
                    display_name="Reasoning Effort",
                    field_type="choice",
                    prompt="Select reasoning effort level (must match model capabilities, see https://platform.openai.com/docs/models)",
                    choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                    default="none",
                    required=False,
                    requires_model=True,  # Shown after model selection
                ),
            ],
        )

    async def list_models(self) -> list[ModelInfo]:
        """
        List available OpenAI models.

        Queries the OpenAI API for available models and filters to GPT-5+ series
        and deep research models.
        Raises exception if API query fails (no fallback - caller handles empty lists).
        """
        # Query OpenAI models API - let exceptions propagate
        models_response = await self.client.models.list()
        models = []

        import re as regex_module

        for model in models_response.data:
            model_id = model.id

            # Check if this is a deep research model
            is_deep_research = model_id in DEEP_RESEARCH_MODELS or model_id.startswith(
                ("o3-deep-research", "o4-mini-deep-research")
            )

            # Filter to GPT-5+ series models or deep research models
            if not (model_id.startswith("gpt-5") or model_id.startswith("gpt-6") or is_deep_research):
                continue

            # Skip dated versions when filtered (e.g., gpt-5-2025-08-07) - duplicates of aliases
            # But always include deep research aliases (o3-deep-research, o4-mini-deep-research)
            if self.filtered and not is_deep_research and regex_module.search(r"-\d{4}-\d{2}-\d{2}$", model_id):
                continue

            # Generate display name from model ID
            display_name = self._model_id_to_display_name(model_id)

            # Determine capabilities based on model type
            if is_deep_research:
                capabilities = ["deep_research", "web_search", "reasoning"]
                context_window = 200000
                max_output_tokens = 100000
                defaults = {"max_tokens": 32768, "background": True}
            else:
                capabilities = ["tools", "reasoning", "streaming", "json_mode"]
                if "mini" in model_id or "nano" in model_id:
                    capabilities.append("fast")
                context_window = 400000
                max_output_tokens = 128000
                defaults = {"max_tokens": 16384, "reasoning_effort": "none"}

            models.append(
                ModelInfo(
                    id=model_id,
                    display_name=display_name,
                    context_window=context_window,
                    max_output_tokens=max_output_tokens,
                    capabilities=capabilities,
                    defaults=defaults,
                )
            )

        # Sort alphabetically by display name
        return sorted(models, key=lambda m: m.display_name.lower())

    def _model_id_to_display_name(self, model_id: str) -> str:
        """Convert model ID to display name with proper capitalization.

        Examples:
            gpt-5.1 -> GPT 5.1
            gpt-5.1-codex -> GPT-5.1 codex
            gpt-5-mini -> GPT-5 mini
            o3-deep-research -> o3 Deep Research
            o4-mini-deep-research -> o4-mini Deep Research
        """
        # Known display name mappings
        display_names = {
            "gpt-5.1": "GPT 5.1",
            "gpt-5.2": "GPT 5.2",
            "gpt-5.2-codex": "GPT 5.2 codex",
            "gpt-5.1-codex": "GPT-5.1 codex",
            "gpt-5.1-codex-max": "GPT-5.1 codex max",
            "gpt-5-mini": "GPT-5 mini",
            "o3-deep-research": "o3 Deep Research",
            "o3-deep-research-2025-06-26": "o3 Deep Research (2025-06-26)",
            "o4-mini-deep-research": "o4-mini Deep Research",
            "o4-mini-deep-research-2025-06-26": "o4-mini Deep Research (2025-06-26)",
        }

        if model_id in display_names:
            return display_names[model_id]

        # Handle deep research model variants
        if "deep-research" in model_id:
            # Extract base model (o3, o4-mini, etc.) and format nicely
            if model_id.startswith("o3-deep-research"):
                suffix = model_id.replace("o3-deep-research", "")
                return f"o3 Deep Research{suffix}"
            if model_id.startswith("o4-mini-deep-research"):
                suffix = model_id.replace("o4-mini-deep-research", "")
                return f"o4-mini Deep Research{suffix}"

        # Generate from ID: capitalize GPT, keep rest lowercase
        if model_id.startswith("gpt-"):
            parts = model_id.split("-", 1)
            if len(parts) == 2:
                return f"GPT-{parts[1]}"
        return model_id

    async def complete(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """
        Generate completion from ChatRequest.

        Args:
            request: Typed chat request with messages, tools, config
            **kwargs: Provider-specific options (override request fields)

        Returns:
            ChatResponse with content blocks, tool calls, usage
        """
        openai_request = self._convert_to_openai_request(request, **kwargs)
        openai_request = self._remove_unused_tool_calls(openai_request)

        request_params = openai_request.model_dump(exclude_none=True)
        await self._emit_request_hooks(request_params)

        start_time = time.time()
        try:
            if self.background:
                response = await self.client.responses.create(
                    model=openai_request.model,
                    input=openai_request.input,
                    include=openai_request.include if openai_request.include is not None else omit,
                    instructions=openai_request.instructions if openai_request.instructions is not None else omit,
                    max_output_tokens=openai_request.max_output_tokens
                    if openai_request.max_output_tokens is not None
                    else omit,
                    parallel_tool_calls=openai_request.parallel_tool_calls
                    if openai_request.parallel_tool_calls is not None
                    else omit,
                    reasoning=openai_request.reasoning if openai_request.reasoning is not None else omit,
                    temperature=openai_request.temperature if openai_request.temperature is not None else omit,
                    text=openai_request.text if openai_request.text is not None else omit,
                    tool_choice=openai_request.tool_choice if openai_request.tool_choice is not None else omit,
                    tools=openai_request.tools if openai_request.tools is not None else omit,
                    truncation=openai_request.truncation if openai_request.truncation is not None else omit,
                    store=True,
                    stream=False,
                    background=self.background,
                )
                while response.status in {"queued", "in_progress"}:
                    await asyncio.sleep(2)
                    response = await self.client.responses.retrieve(response.id)
            else:
                response_stream = await self.client.responses.create(
                    model=openai_request.model,
                    input=openai_request.input,
                    include=openai_request.include if openai_request.include is not None else omit,
                    instructions=openai_request.instructions if openai_request.instructions is not None else omit,
                    max_output_tokens=openai_request.max_output_tokens
                    if openai_request.max_output_tokens is not None
                    else omit,
                    parallel_tool_calls=openai_request.parallel_tool_calls
                    if openai_request.parallel_tool_calls is not None
                    else omit,
                    reasoning=openai_request.reasoning if openai_request.reasoning is not None else omit,
                    temperature=openai_request.temperature if openai_request.temperature is not None else omit,
                    text=openai_request.text if openai_request.text is not None else omit,
                    tool_choice=openai_request.tool_choice if openai_request.tool_choice is not None else omit,
                    tools=openai_request.tools if openai_request.tools is not None else omit,
                    truncation=openai_request.truncation if openai_request.truncation is not None else omit,
                    store=False,
                    stream=True,
                    background=False,
                )
                response: Response | None = None
                async for event in response_stream:
                    if isinstance(event, ResponseCompletedEvent):
                        response = event.response
                        break
                if response is None:
                    raise RuntimeError("Response stream ended without a completed response event.")

            elapsed_ms = int((time.time() - start_time) * 1000)

            await self._emit_response_hooks(response, elapsed_ms)

        except openai.BadRequestError as e:
            if e.code == "context_length_exceeded":
                raise ContextLimitExceededError(str(e)) from e
            raise

        final_response = self._convert_from_openai_response(response)
        return final_response

    def _convert_to_openai_request(self, request: ChatRequest, **kwargs: Any) -> OpenAIRequest:
        """Converts Amplifier Core's ChatRequest to OpenAIRequest which wraps OpenAI's Responses API parameters."""
        messages = self._convert_to_response_input_item_param(request.messages)

        # Get model from kwargs (passed by orchestrator) or fall back to default
        model = kwargs.get("model", self.default_model)

        openai_request = OpenAIRequest(
            input=messages,
            model=model,
            include=["reasoning.encrypted_content"],
            reasoning={"effort": self.reasoning, "summary": "auto"},
            tools=self._convert_to_tool_param(request.tools) if request.tools else None,
            max_output_tokens=request.max_output_tokens,
            truncation="auto",
            background=False,
        )
        return openai_request

    def _convert_to_response_input_item_param(self, messages: list[Message]) -> list[ResponseInputItemParam]:
        """
        if role = "user": only handle TextBlock and ImageBlock, or str
          - str -> EasyInputMessageParam
          - TextBlock and ImageBlock -> EasyInputMessageParam with ResponseInputMessageContentListParam

        if role = "system" or "developer": only handle TextBlock or str
          - str -> EasyInputMessageParam
          - TextBlock -> concat all text parts into single content str in EasyInputMessageParam

        if role = "assistant": handle ThinkingBlock, TextBlock, and ToolCallBlock, or str
          - str -> EasyInputMessageParam
          - ThinkingBlock -> ResponseReasoningItemParam. Place first, if exists. Only include if metadata `created_by` = "openai"
            - OpenAI does not support thinking from other providers.
          - TextBlock -> collect all text blocks, aggregate into a single string in EasyInputMessageParam. Must go before any ResponseFunctionToolCallParam
          - ToolCallBlock -> ResponseFunctionToolCallParam

        if role = "tool": only handle str or ToolResultBlock
          - str -> FunctionCallOutput
          - ToolResultBlock -> FunctionCallOutput
        """
        response_input_item_param: list[ResponseInputItemParam] = []
        for message in messages:
            match message.role:
                case "user":
                    if isinstance(message.content, str):
                        response_input_item_param.append(EasyInputMessageParam(role="user", content=message.content))
                    elif isinstance(message.content, list):
                        content_params: list[ResponseInputContentParam] = []
                        for content_block in message.content:
                            match content_block.type:
                                case "text":
                                    content_params.append(
                                        ResponseInputTextParam(type="input_text", text=content_block.text)
                                    )
                                case "image":
                                    content_params.append(
                                        ResponseInputImageParam(
                                            type="input_image", image_url="NOT IMPLEMENTED YET", detail="auto"
                                        )
                                    )
                        response_input_item_param.append(EasyInputMessageParam(role="user", content=content_params))
                case "system" | "developer":
                    if isinstance(message.content, str):
                        response_input_item_param.append(
                            EasyInputMessageParam(role=message.role, content=message.content)
                        )
                    elif isinstance(message.content, list):
                        aggregated_text = ""
                        for content_block in message.content:
                            match content_block.type:
                                case "text":
                                    aggregated_text += content_block.text + "\n"
                        aggregated_text = aggregated_text.strip()
                        response_input_item_param.append(
                            EasyInputMessageParam(role=message.role, content=aggregated_text)
                        )
                case "assistant":
                    assistant_items: list[ResponseInputItemParam] = []
                    if isinstance(message.content, str):
                        assistant_items.append(EasyInputMessageParam(role="assistant", content=message.content))
                    elif isinstance(message.content, list):
                        metadata = message.metadata or {}
                        created_by = metadata.get("created_by", "")
                        for content_block in message.content:
                            match content_block.type:
                                case "thinking":
                                    if content_block.content is None:
                                        continue
                                    encrypted_content = content_block.content[0]
                                    reasoning_id = content_block.content[1]
                                    if created_by == "openai":
                                        assistant_items.append(
                                            ResponseReasoningItemParam(
                                                id=reasoning_id,
                                                type="reasoning",
                                                summary=[{"type": "summary_text", "text": content_block.thinking}],
                                                encrypted_content=encrypted_content,
                                            )
                                        )
                                case "text":
                                    aggregated_text = content_block.text
                                    assistant_items.append(
                                        EasyInputMessageParam(role="assistant", content=aggregated_text)
                                    )
                                case "tool_call":
                                    assistant_items.append(
                                        ResponseFunctionToolCallParam(
                                            arguments=json.dumps(content_block.input),
                                            call_id=content_block.id,
                                            name=content_block.name,
                                            type="function_call",
                                        )
                                    )
                    assistant_items = self._reorder_input_item_params(assistant_items)
                    response_input_item_param.extend(assistant_items)
                case "tool" | "function":
                    if isinstance(message.content, str) and message.tool_call_id is not None:
                        function_call_output = FunctionCallOutput(
                            call_id=message.tool_call_id,
                            output=message.content,
                            type="function_call_output",
                        )
                        response_input_item_param.append(function_call_output)
                    elif isinstance(message.content, list):
                        for content_block in message.content:
                            match content_block.type:
                                case "tool_result":
                                    function_call_output = FunctionCallOutput(
                                        call_id=content_block.tool_call_id,
                                        output=str(content_block.output),
                                        type="function_call_output",
                                    )
                                    response_input_item_param.append(function_call_output)

        return response_input_item_param

    def _reorder_input_item_params(self, items: list[ResponseInputItemParam]) -> list[ResponseInputItemParam]:
        """
        Re-order input items params so that
        - Any ResponseReasoningItemParam come first
        - ToolCallBlocks come last
        """
        reasoning_items: list[ResponseInputItemParam] = []
        tool_call_items: list[ResponseInputItemParam] = []
        other_items: list[ResponseInputItemParam] = []

        for item in items:
            item_type = item.get("type") if isinstance(item, dict) else None
            if item_type == "reasoning":
                reasoning_items.append(item)
            elif item_type == "function_tool_call":
                tool_call_items.append(item)
            else:
                other_items.append(item)

        return reasoning_items + other_items + tool_call_items

    def _remove_unused_tool_calls(self, request: OpenAIRequest) -> OpenAIRequest:
        """
        Remove any tool calls from the request that do not have a corresponding tool result,
        maintaining original order of input messages.
        """
        # Collect all call_ids that have corresponding results
        result_call_ids: set[str] = set()
        for item in request.input:
            if isinstance(item, dict) and item.get("type") == "function_call_output":
                call_id = item.get("call_id")
                if call_id:
                    result_call_ids.add(call_id)

        # Filter out tool calls without matching results
        filtered_input: list[ResponseInputItemParam] = []
        for item in request.input:
            if isinstance(item, dict) and item.get("type") == "function_call":
                call_id = item.get("call_id")
                if call_id not in result_call_ids:
                    continue  # Skip tool calls without results
            filtered_input.append(item)

        request.input = filtered_input
        return request

    def _convert_to_tool_param(self, tools: list[ToolSpec]) -> list[ToolParam]:
        """Convert Amplifier ToolSpec list to OpenAI ToolParam list."""
        result: list[ToolParam] = []
        for tool in tools:
            # Deep copy and make strict-mode compliant recursively
            parameters = self._make_schema_strict(tool.parameters.copy())

            tool_param: FunctionToolParam = {
                "type": "function",
                "name": tool.name,
                "parameters": parameters,
                "strict": True,
            }
            if tool.description is not None:
                tool_param["description"] = tool.description
            result.append(tool_param)
        return result

    def _make_schema_strict(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Recursively make a JSON schema strict-mode compliant for OpenAI.

        OpenAI's strict mode requires:
        - "additionalProperties": false on ALL objects (even without properties)
        - "required" must list ALL properties (no optional fields)
        - Objects without properties need empty "properties": {} and "required": []

        This function recursively traverses the schema and applies these rules
        to all nested objects, array items, anyOf/oneOf variants, etc.
        """
        if not isinstance(schema, dict):
            return schema

        # Deep copy to avoid mutating the original
        schema = {k: v for k, v in schema.items()}

        # If this is an object type, make it strict (even without properties)
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            if "properties" in schema:
                # All properties must be required in strict mode
                schema["required"] = list(schema["properties"].keys())
                # Recursively process each property
                schema["properties"] = {
                    key: self._make_schema_strict(prop) for key, prop in schema["properties"].items()
                }
            else:
                # Object without properties - add empty properties and required
                schema["properties"] = {}
                schema["required"] = []

        # Handle array items
        if "items" in schema:
            schema["items"] = self._make_schema_strict(schema["items"])

        # Handle anyOf, oneOf, allOf
        for key in ("anyOf", "oneOf", "allOf"):
            if key in schema:
                schema[key] = [self._make_schema_strict(variant) for variant in schema[key]]

        # Handle $defs / definitions
        for key in ("$defs", "definitions"):
            if key in schema:
                schema[key] = {name: self._make_schema_strict(defn) for name, defn in schema[key].items()}

        return schema

    def _convert_from_openai_response(self, openai_response: Response) -> ChatResponse:
        content_blocks: list[ContentBlockUnion] = []
        tool_calls: list[ToolCall] = []
        metadata = {"created_by": "openai", "raw_response": openai_response.model_dump(mode="json")}
        name = None
        tool_call_id = None
        for message in openai_response.output:
            match message.type:
                case "reasoning":
                    summary = [x.text for x in message.summary if message.summary is not None]
                    summary = "\n".join(summary)
                    thinking_block = ThinkingBlock(
                        type="thinking",
                        thinking=summary,
                        signature=None,
                        content=[message.encrypted_content, message.id],
                    )
                    content_blocks.append(thinking_block)
                case "message":
                    for content in message.content:
                        match content.type:
                            case "output_text":
                                text_block = TextBlock(type="text", text=content.text)
                                content_blocks.append(text_block)
                case "function_call":
                    tool_call_id = message.call_id
                    name = message.name
                    arguments = json.loads(message.arguments)
                    tool_call_obj = ToolCall(
                        id=tool_call_id,
                        name=name,
                        arguments=arguments,
                    )
                    tool_call_block = ToolCallBlock(
                        type="tool_call",
                        id=tool_call_id,
                        name=name,
                        input=arguments,
                    )
                    content_blocks.append(tool_call_block)
                    tool_calls.append(tool_call_obj)

        if openai_response.usage is not None:
            usage = Usage(
                input_tokens=openai_response.usage.input_tokens,
                output_tokens=openai_response.usage.output_tokens,
                total_tokens=openai_response.usage.total_tokens,
            )
        else:
            usage = None
        # Responses API does not support finish_reason
        chat_response = ChatResponse(
            content=content_blocks,
            tool_calls=tool_calls,
            usage=usage,
            metadata=metadata,
        )
        return chat_response

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """
        Parse tool calls from ChatResponse.

        Args:
            response: Typed chat response

        Returns:
            List of tool calls to execute
        """
        return response.tool_calls or []

    def _truncate_values(self, obj: Any, max_length: int | None = None) -> Any:
        """Recursively truncate string values in nested structures.

        Preserves structure, only truncates leaf string values longer than max_length.
        Uses self.debug_truncate_length if max_length not specified.

        Args:
            obj: Any JSON-serializable structure (dict, list, primitives)
            max_length: Maximum string length (defaults to self.debug_truncate_length)

        Returns:
            Structure with truncated string values
        """
        if max_length is None:
            max_length = self.debug_truncate_length

        # Type guard: max_length is guaranteed to be int after this point
        assert isinstance(max_length, int), "debug_truncate_length must be an int"

        if isinstance(obj, str):
            if len(obj) > max_length:
                return obj[:max_length] + f"... (truncated {len(obj) - max_length} chars)"
            return obj
        if isinstance(obj, dict):
            return {k: self._truncate_values(v, max_length) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._truncate_values(item, max_length) for item in obj]
        return obj  # Numbers, booleans, None pass through unchanged

    async def _emit_request_hooks(self, params: dict[str, Any]) -> None:
        """Emit request-related hook events.

        Emits:
            - llm:request: Summary info (always)
            - llm:request:debug: Full request with truncated values (if debug enabled)
            - llm:request:raw: Complete untruncated request (if debug AND raw_debug enabled)
        """
        if not self.coordinator or not hasattr(self.coordinator, "hooks"):
            return

        # Count messages for summary
        input_items = params.get("input", [])
        message_count = len(list(input_items)) if input_items else 0
        tools = params.get("tools")
        tool_count = len(list(tools)) if tools else 0

        # INFO level: Request summary (always emitted)
        await self.coordinator.hooks.emit(
            "llm:request",
            {
                "lvl": "INFO",
                "provider": self.name,
                "model": params.get("model"),
                "message_count": message_count,
                "tool_count": tool_count,
                "reasoning": params.get("reasoning"),
            },
        )

        # DEBUG level: Full request payload with truncated values (if debug enabled)
        if self.debug:
            await self.coordinator.hooks.emit(
                "llm:request:debug",
                {
                    "lvl": "DEBUG",
                    "provider": self.name,
                    "request": self._truncate_values(params),
                },
            )

        # RAW level: Complete params dict as sent to OpenAI API (if debug AND raw_debug enabled)
        if self.debug and self.raw_debug:
            await self.coordinator.hooks.emit(
                "llm:request:raw",
                {
                    "lvl": "DEBUG",
                    "provider": self.name,
                    "params": params,  # Complete untruncated params
                },
            )

    async def _emit_response_hooks(self, response: Response, elapsed_ms: int) -> None:
        """Emit response-related hook events.

        Emits:
            - llm:response: Summary info (always)
            - llm:response:debug: Full response with truncated values (if debug enabled)
            - llm:response:raw: Complete untruncated response (if debug AND raw_debug enabled)
        """
        if not self.coordinator or not hasattr(self.coordinator, "hooks"):
            return

        # Extract usage info
        usage_info = None
        if response.usage:
            usage_info = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        # INFO level: Response summary (always emitted)
        await self.coordinator.hooks.emit(
            "llm:response",
            {
                "lvl": "INFO",
                "provider": self.name,
                "model": response.model,
                "usage": usage_info,
                "status": getattr(response, "status", "unknown"),
                "duration_ms": elapsed_ms,
            },
        )

        # Convert response to dict for debug/raw emissions
        response_dict = response.model_dump()

        # DEBUG level: Full response with truncated values (if debug enabled)
        if self.debug:
            await self.coordinator.hooks.emit(
                "llm:response:debug",
                {
                    "lvl": "DEBUG",
                    "provider": self.name,
                    "response": self._truncate_values(response_dict),
                    "status": "ok",
                    "duration_ms": elapsed_ms,
                },
            )

        # RAW level: Complete response object from OpenAI API (if debug AND raw_debug enabled)
        if self.debug and self.raw_debug:
            await self.coordinator.hooks.emit(
                "llm:response:raw",
                {
                    "lvl": "DEBUG",
                    "provider": self.name,
                    "response": response_dict,  # Complete untruncated response
                },
            )
