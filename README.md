# Amplifier OpenAI Provider Module

OpenAI model integration for Amplifier via the Responses API.


## Prerequisites

- **Python 3.11+**
- **[uv](https://github.com/astral-sh/uv)**


## Configuration

```yaml
providers:
  - module: provider-openai-v2
    config:
      base_url: null                    # Optional custom endpoint (null = OpenAI default)
      default_model: gpt-5.2-codex
      max_tokens: 64000
      reasoning: high                   # Reasoning effort: minimal|low|medium|high|xhigh
      filtered: true                    # Filter to curated model list
      web_search: true                  # Enable built-in web search tool
      timeout: 1200                     # Request timeout in seconds (default 20 min, 0 to disable)
      background: false                 # Background mode for very long requests
      debug: false                      # Enable debug events
      raw_debug: false                  # Enable raw API I/O logging
```

- Recommended models are:
  - `gpt-5.2-codex` for code tasks
  - `gpt-5.2` for general tasks
  - `gpt-5.2-pro` for complex reasoning where latency is not critical
- Background mode (`background: true`) offloads the request to OpenAI's async pipeline; try this for very long-running requests that may exceed HTTP timeout limits.


### Debug Configuration

Standard debug (`debug: true`):
- Emits `llm:request:debug` and `llm:response:debug` events
- Contains request/response summaries with message counts, model info, usage stats

Raw debug (`debug: true, raw_debug: true`):
- Emits `llm:request:raw` and `llm:response:raw` events
- Contains complete, unmodified request params and response objects
- Use only for deep provider integration debugging

## Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Development

### Setup

```bash
# Install dependencies (including dev tools)
uv sync --all-extras --all-groups
```

### Code Quality Checks

This module uses [Ruff](https://docs.astral.sh/ruff/) for formatting and linting, and [Pyright](https://github.com/microsoft/pyright) for type checking.

```bash
# Lint and auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Type check
uv run pyright
```

### Running Tests

```bash
uv run pytest
```


## Limitations

- Image support (TBD, limited ecosystem support)
- o-series models — prefer gpt-5.x or later
- Tool repair — incomplete tool calls are removed instead
- Temperature and similar parameters — not supported by Responses API
- `store` parameter — always set to `False` unless background mode is enabled (which requires `store=True`)


## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
