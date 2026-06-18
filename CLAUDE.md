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
uv run python run_server.py
# Or directly:
uv run uvicorn travel_agent.api.server:create_app --factory --host 127.0.0.1 --port 8000 --reload

# Run CLI mode
uv run python run_cli.py

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

### Package structure (refactored 2026-06-19)

```
src/travel_agent/
├── config.py               Pydantic config (unchanged)
├── api/          (9 files) Web & CLI layer (FastAPI, WebSocket, A2UI, maps)
├── agent/        (7 files) Agent assembly (factory, context, adapter, precheck)
├── tools/        (19 files) Core tools: search/ planning/ rendering/ utility/
├── mcp/          (6 files) MCP server + tool registration + interceptors
├── orchestration/ (6 files) 5-layer pipeline (opt-in)
├── storage/      (5 files) L1/L2/L3 memory system
├── skills/       (1 file)  Markdown skill loading
└── utils/        (3 files) Prompt builder, logging
```

### Dual-server design

FastAPI (`api/server.py`) and MCP Server (`mcp/server.py`) are two separate processes. FastAPI's `lifespan` starts the MCP Server as an `asyncio.Task` on port 8002. This is not optional — the agent cannot function without the MCP Server running.

The dependency chain:
- `build_agent()` in `agent/factory.py` connects to the MCP Server via `MultiServerMCPClient` to fetch all tools
- The ReAct agent calls tools through the MCP client, not by importing tool functions directly
- All 16 core tools are registered on the MCP Server in `mcp/register_tools.py`

### Entry points

Two parallel entry paths share the same `build_agent()` → `ClientContext` pipeline:
1. **WebSocket** (`api/websocket_handler.py`) — browser SPA, auto-extracts map/weather JSON blocks from tool results
2. **CLI** (`api/cli.py`) — stdin/stdout loop, same agent but simpler output

Launchers: `python run_server.py` / `python run_cli.py`

### DeepSeek content quirk

DeepSeek's API rejects `content` fields that are lists. Two places handle this:
- `DeepSeekChatOpenAI._get_request_payload()` in `agent/deepseek_adapter.py` — flattens content before sending
- `clean_messages_for_next_turn()` in `api/message_utils.py` — serializes list/dict content for history reuse

### Three-layer memory system

Orchestrated by `agent/context.py` — `ClientContext.prepare_messages_for_invoke()`:
- **L1** (`storage/memory_compressor.py`): LLM summarization of older messages
- **L2** (`storage/agent_memory.py`): Per-session tool result persistence
- **L3** (`storage/user_profile.py`): Cross-session user preferences

Strategy selection via `agent/memory_switch.py`: `full_context` / `compressed_context` / `profile_only`.

### Layered orchestration (opt-in)

When `[orchestration].enabled = true`, `build_agent()` wraps the ReAct agent in `LayeredTravelAgent` (`orchestration/layered_agent.py`) which executes tools in a 5-layer sequence: requirement → research → planning → risk → render. Each layer exposes a subset of tools (defined in `agent/node_manager.py`). Scenario tags are inferred from `orchestration/scenario_tags.py` (shared between LayeredTravelAgent and NodeManager).

### Skills vs tools

Skills (`.storyline/skills/*/SKILL.md`) are Markdown-defined workflows loaded via `skillkit`. They appear as additional tools to the LLM but describe multi-step strategies. Both core tools and skills are merged in `build_agent()`.

### Prompt management

Prompts in `prompts/tasks/<task>/<lang>/*.md`. Loaded by `utils/prompts.py` — `PromptBuilder` with `{{variable}}` placeholders. Global system prompt: `prompts/tasks/instruction/zh/system.md`.

### Config loading

`config.py` — Pydantic `Settings` model, loads `config.toml`. Relative paths resolved relative to config file directory. Override with `TRAVEL_AGENT_CONFIG` env var.

### Frontend data protocol

WebSocket appends JSON blocks (```json with `__type` field) to LLM's reply. Frontend parses `pois`/`itinerary`/`route`/`weather` types. Extraction in `api/map_extractor.py` includes a fallback that auto-generates itinerary when LLM forgets to call `smart_plan_itinerary`.
