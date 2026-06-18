"""
MCP Tool 调用前后的轻量 Hook。

这些函数用于在工具执行前后统一记录 session、耗时等横切信息。
它们是普通包装函数，并不是 FastMCP 官方中间件。

当前 ``register_tools.py`` 已在各 Wrapper 中直接完成结果持久化，
因此 after hook 只记录耗时并原样返回结果，后续可扩展 tracing/metrics。
"""
from __future__ import annotations

import time
from typing import Any, Callable, Awaitable

from travel_agent.storage.agent_memory import ArtifactStore

try:
    from travel_agent.utils.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)


async def before_tool_call(
    tool_name: str,
    session_id: str,
    store: ArtifactStore,
    **kwargs: Any,
) -> dict:
    """
    在工具执行前创建调用上下文。

    返回值会继续传给 ``after_tool_call``，用于计算耗时和关联日志。
    """
    logger.debug("[Hook:before] tool=%s session=%s", tool_name, session_id)
    return {"tool_name": tool_name, "session_id": session_id, "start_ts": time.time()}


async def after_tool_call(
    result: Any,
    ctx: dict,
    store: ArtifactStore,
) -> Any:
    """
    在工具执行后记录耗时并返回结果。

    若结果包含 ``isError=False``、``result`` 等 MCP 信封字段，说明
    ``register_tools`` 已完成持久化，此处不应重复写 Artifact。
    """
    elapsed = time.time() - ctx.get("start_ts", time.time())
    logger.debug(
        "[Hook:after] tool=%s session=%s elapsed=%.2fs",
        ctx.get("tool_name"),
        ctx.get("session_id"),
        elapsed,
    )
    return result


def wrap_tool(
    tool_fn: Callable[..., Awaitable[Any]],
    tool_name: str,
    session_id: str,
    store: ArtifactStore,
) -> Callable[..., Awaitable[Any]]:
    """
    为异步工具套用 before/after hook，并保持原函数名称和文档。

    示例::

        wrapped = wrap_tool(my_tool_fn, "search_poi", session_id, store)
        result = await wrapped(**kwargs)
    """
    async def _wrapped(**kwargs: Any) -> Any:
        ctx = await before_tool_call(
            tool_name=tool_name,
            session_id=session_id,
            store=store,
            **kwargs,
        )
        result = await tool_fn(**kwargs)
        return await after_tool_call(result, ctx, store)

    _wrapped.__name__ = tool_fn.__name__
    _wrapped.__doc__ = tool_fn.__doc__
    return _wrapped
