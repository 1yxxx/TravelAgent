# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Install deps (uses uv, not pip directly)
uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt
# Or use the helper script:
bash build_env.sh

# Copy and edit config (config.toml is gitignored, contains real API keys)
cp config.toml.example config.toml

# Run web server (port 8000; MCP server auto-starts on port 8002 via lifespan)
uv run uvicorn agent_fastapi:app --host 127.0.0.1 --port 8000 --reload

# Run CLI mode
uv run python cli.py

# Run tests (no special config needed)
uv run pytest

# Run a single test file
uv run pytest tests/test_precheck.py -v

# Validate API keys
uv run python scripts/validate_api_keys.py

# Evaluate layered orchestration metrics
uv run python scripts/eval_layer_metrics.py --outputs-dir travel_outputs --out-dir travel_outputs/eval
```

## Architecture

### Dual-server design

FastAPI (`agent_fastapi.py`) and MCP Server (`src/travel_agent/mcp/server.py`) are two separate processes. FastAPI's `lifespan` starts the MCP Server as an `asyncio.Task` on port 8002. This is not optional — the agent cannot function without the MCP Server running.

The dependency chain:
- `build_agent()` in `agent.py` connects to the MCP Server via `MultiServerMCPClient` to fetch all tools
- The ReAct agent calls tools through the MCP client, not by importing tool functions directly
- All 16 tools are registered on the MCP Server in `register_tools.py`

### Entry points

Two parallel entry paths share the same `build_agent()` → `ClientContext` pipeline:
1. **WebSocket** (`agent_fastapi.py:/ws/chat`) — browser SPA, SSE-style streaming, auto-extracts map/weather JSON blocks from tool results
2. **CLI** (`cli.py`) — stdin/stdout loop, same agent but simpler output

### DeepSeek content quirk

DeepSeek's API rejects `content` fields that are lists (standard OpenAI format allows them). Two places handle this:
- `DeepSeekChatOpenAI._get_request_payload()` in `agent.py` — flattens content before sending to the API
- `_clean_messages_for_next_turn()` in `agent_fastapi.py` — serializes list/dict content to strings for message history reuse

The `_flatten_content()` / `_normalize_content()` helpers convert list-type content to strings. If adding a new model provider, check whether it tolerates list content.

### Three-layer memory system

`ClientContext.prepare_messages_for_invoke()` orchestrates memory based on message count and token estimate:
- **L1** (`MemoryCompressor`): LLM-based summarization of older messages, triggered at soft thresholds
- **L2** (`ArtifactStore`): Session-scoped persistence of all tool call results to disk; `build_context_prompt()` injects a snapshot into the system prompt so the LLM knows what data has already been collected
- **L3** (`UserProfileStore`): Cross-session user preference extraction (budget, pace, interests), injected into system prompt

The `choose_memory_framework()` function picks between `full_context` / `compressed_context` / `profile_only` modes based on configurable thresholds in `config.toml` `[memory_switch]`.

### Layered orchestration (opt-in)

When `[orchestration].enabled = true`, `build_agent()` wraps the ReAct agent in `LayeredTravelAgent` which executes tools in a fixed 5-layer sequence: requirement → research → planning → risk → render. Each layer only exposes a subset of tools (defined in `NodeManager`). Failure triggers rollback to the last passing checkpoint + retry.

`NodeManager.grouped_tools_for_messages()` also does scenario-based tool filtering (family/senior/couple/solo/luxury/budget/short_trip/long_trip) inferred from message text.

### Skills vs tools

Skills (`.storyline/skills/*/SKILL.md`) are Markdown-defined composite workflows loaded via `skillkit`. They appear as additional tools to the LLM but describe multi-step strategies rather than single API calls. Core tools come from the MCP Server; skills come from the filesystem. Both are merged into one tool list in `build_agent()`.

### Prompt management

All prompts live in `prompts/tasks/<task>/<lang>/system.md` and `user.md`. They are loaded at runtime by `PromptBuilder`, support `{{variable}}` placeholders, and are cached in memory after first load. The global system prompt for the agent is in `prompts/tasks/instruction/zh/system.md`.

### Config loading

`Settings` (Pydantic model in `config.py`) loads `config.toml`. All relative paths in config are resolved relative to the config file's directory, not CWD. Override the config path with env var `TRAVEL_AGENT_CONFIG`.

### Frontend data protocol

The WebSocket endpoint appends JSON blocks (```json fences with `__type` field) to the LLM's text response. The frontend (`web/static/app.js`) parses these to render POI markers, itinerary cards, route polylines, and weather widgets on the AMap. Three block types: `pois`, `itinerary`, `route`. Extraction logic in `_extract_map_blocks()` also includes a fallback that auto-generates an itinerary block when the LLM searches for POIs but forgets to call `smart_plan_itinerary`.

### Dependencies on the OpenStoryline framework

This project is built on [FireRed-OpenStoryline](https://github.com/FireRedTeam/FireRed-OpenStoryline). The `skillkit` library, `ArtifactStore`, `SessionLifecycleManager`, and the MCP tool registration pattern come from that framework. Don't refactor these without understanding their upstream conventions.
