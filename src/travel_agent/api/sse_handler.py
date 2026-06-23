"""
基于 HTTP POST + Server-Sent Events 的聊天处理器。

客户端通过 POST 提交用户消息，服务端在同一个 HTTP 响应中持续发送 token、
工具状态、A2UI、地图和天气事件。客户端断开后，Starlette 会取消本生成器，
进而取消正在运行的 Agent task。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any, AsyncIterator, Dict, Literal, Optional

from fastapi import Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from travel_agent.api.a2ui_bridge import a2ui_payload_if_enabled, extract_place_cards
from travel_agent.api.chat_session import ChatSessionStore
from travel_agent.api.map_extractor import (
    build_call_id_to_name,
    extract_map_blocks,
    extract_weather_block,
)
from travel_agent.api.message_utils import clean_messages_for_next_turn, normalize_content
from travel_agent.config import load_settings
from travel_agent.utils.logging import logger


AGENT_TIMEOUT = 120
MAX_RETRIES = 1
AGENT_RECURSION_LIMIT = 20
HEARTBEAT_SECONDS = 15


class ChatStreamRequest(BaseModel):
    """POST /api/chat/stream 的请求体。"""

    session_id: Optional[str] = Field(default=None, max_length=128)
    type: Literal["message", "form_response"] = "message"
    content: str = Field(default="", max_length=20_000)
    data: Optional[Dict[str, Any]] = None


SESSION_STORE = ChatSessionStore()


def encode_sse(event: str, data: Any) -> str:
    """编码标准 SSE 帧；JSON 中的换行会被转义，不会破坏帧边界。"""
    safe_event = re.sub(r"[^a-zA-Z0-9_-]", "_", event) or "message"
    payload = json.dumps(
        data if data is not None else {},
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    return f"event: {safe_event}\ndata: {payload}\n\n"


def create_sse_response(
    stream: AsyncIterator[str],
) -> StreamingResponse:
    """创建禁用代理缓冲的 SSE 响应。"""
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _new_session_id() -> str:
    return f"travel_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def _form_response_to_text(data: Optional[Dict[str, Any]]) -> str:
    """把 A2UI 表单响应转换成 Agent 能理解的用户消息。"""
    fields = data or {}
    parts = []
    if fields.get("destination"):
        parts.append(f"目的地：{fields['destination']}")
    if fields.get("days"):
        parts.append(f"天数：{fields['days']}天")
    if fields.get("budget"):
        labels = {
            "budget": "经济实惠",
            "mid": "舒适享受",
            "luxury": "豪华体验",
        }
        parts.append(f"预算：{labels.get(fields['budget'], fields['budget'])}")
    if fields.get("preference"):
        parts.append(f"偏好：{fields['preference']}")
    return "，".join(parts) + "。请帮我规划旅行。" if parts else ""


def _extract_chunk_text(chunk: Any) -> str:
    if chunk is None:
        return ""
    content = getattr(chunk, "content", chunk if isinstance(chunk, str) else "")
    return normalize_content(content) if content else ""


def _event_value(value: Any, *, limit: int = 2_000) -> Any:
    """限制工具事件体积，并把不可 JSON 序列化对象转换为字符串。"""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        text = str(value)
    if len(text) > limit:
        return text[:limit] + "…"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def _produce_agent_events(
    agent: Any,
    invoke_messages: list,
    queue: asyncio.Queue,
) -> None:
    """执行一次 Agent，并把模型/工具事件写入队列。"""
    config = {"recursion_limit": AGENT_RECURSION_LIMIT}
    result: Optional[Dict[str, Any]] = None

    try:
        async with asyncio.timeout(AGENT_TIMEOUT):
            stream_method = getattr(agent, "astream_events", None)
            if callable(stream_method):
                latest_result: Optional[Dict[str, Any]] = None
                async for event in stream_method(
                    {"messages": invoke_messages},
                    config=config,
                    version="v2",
                ):
                    event_name = event.get("event", "")
                    event_data = event.get("data") or {}

                    if event_name == "on_chat_model_stream":
                        text = _extract_chunk_text(event_data.get("chunk"))
                        if text:
                            await queue.put(("event", "token", {"content": text}))
                    elif event_name == "on_tool_start":
                        await queue.put((
                            "event",
                            "tool_start",
                            {
                                "name": event.get("name", ""),
                                "run_id": str(event.get("run_id", "")),
                                "input": _event_value(event_data.get("input")),
                            },
                        ))
                    elif event_name == "on_tool_end":
                        await queue.put((
                            "event",
                            "tool_end",
                            {
                                "name": event.get("name", ""),
                                "run_id": str(event.get("run_id", "")),
                                "output": _event_value(event_data.get("output")),
                            },
                        ))
                    elif event_name == "on_chain_end":
                        output = event_data.get("output")
                        if isinstance(output, dict) and isinstance(output.get("messages"), list):
                            latest_result = output

                if latest_result is None:
                    raise RuntimeError("Agent 事件流结束，但没有返回最终 messages")
                result = latest_result
            else:
                # 自定义分层编排器当前只实现 ainvoke，保持功能并降级为整段 token。
                result = await agent.ainvoke(
                    {"messages": invoke_messages},
                    config=config,
                )

        await queue.put(("result", result))
    except BaseException as exc:  # task 的异常必须传回响应生成器
        await queue.put(("error", exc))


async def _run_attempt(
    request: Request,
    agent: Any,
    invoke_messages: list,
) -> AsyncIterator[tuple[str, Any]]:
    """桥接 Agent task 与 SSE 生成器，并在无数据期间发送心跳。"""
    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(_produce_agent_events(agent, invoke_messages, queue))
    try:
        while True:
            if await request.is_disconnected():
                raise asyncio.CancelledError
            try:
                item = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                yield "heartbeat", None
                continue

            kind = item[0]
            if kind == "event":
                yield "event", (item[1], item[2])
                continue
            if kind == "result":
                yield "result", item[1]
                return
            if kind == "error":
                raise item[1]
    finally:
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


async def stream_chat(
    request: Request,
    payload: ChatStreamRequest,
    config_path: str,
    *,
    store: ChatSessionStore = SESSION_STORE,
) -> AsyncIterator[str]:
    """处理一轮对话并输出 SSE 文本帧。"""
    session_id = (payload.session_id or "").strip() or _new_session_id()
    yield encode_sse("session", {"session_id": session_id})

    if payload.type == "form_response":
        user_text = _form_response_to_text(payload.data)
    else:
        user_text = payload.content.strip()

    if not user_text:
        yield encode_sse("error", {"message": "消息内容不能为空"})
        yield encode_sse("done", {"ok": False})
        return

    cfg = load_settings(config_path)
    session = await store.get_or_create(session_id)
    user_message: Optional[HumanMessage] = None
    committed = False

    try:
        async with session.lock:
            session.last_access = time.monotonic()
            try:
                await store.ensure_initialized(session, cfg)
            except Exception as exc:  # noqa: BLE001
                logger.exception("[session=%s] build_agent 失败: %s", session_id, exc)
                yield encode_sse("error", {"message": f"后端初始化失败：{exc}"})
                yield encode_sse("done", {"ok": False})
                return

            user_message = HumanMessage(content=user_text)
            session.messages.append(user_message)

            invoke_messages, memory_meta = (
                await session.context.prepare_messages_for_invoke(session.messages)
            )
            yield encode_sse("memory_mode", {
                "mode": memory_meta.get("mode"),
                "reason": memory_meta.get("reason"),
                "message_count": memory_meta.get("message_count"),
                "token_estimate": memory_meta.get("token_estimate"),
            })

            result: Optional[Dict[str, Any]] = None
            last_exc: Optional[BaseException] = None
            streamed_text_parts: list[str] = []

            for attempt in range(1, MAX_RETRIES + 2):
                attempt_had_tokens = False
                try:
                    async for kind, value in _run_attempt(
                        request,
                        session.agent,
                        invoke_messages,
                    ):
                        if kind == "heartbeat":
                            yield ": heartbeat\n\n"
                        elif kind == "event":
                            event_name, event_data = value
                            if event_name == "token":
                                attempt_had_tokens = True
                                streamed_text_parts.append(event_data["content"])
                            yield encode_sse(event_name, event_data)
                        elif kind == "result":
                            result = value
                    last_exc = None
                    break
                except asyncio.CancelledError:
                    logger.info("[session=%s] SSE 客户端断开，取消 Agent", session_id)
                    raise
                except BaseException as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning(
                        "[session=%s] Agent 调用失败（第 %d 次）: %s",
                        session_id,
                        attempt,
                        exc,
                    )
                    if attempt <= MAX_RETRIES:
                        if attempt_had_tokens:
                            streamed_text_parts.clear()
                        yield encode_sse("retry", {
                            "attempt": attempt,
                            "reason": "timeout" if isinstance(exc, TimeoutError) else str(exc),
                            "reset": attempt_had_tokens,
                        })

            if result is None:
                error_text = str(last_exc) if last_exc else "未知错误"
                if session.messages and session.messages[-1] is user_message:
                    session.messages.pop()
                yield encode_sse("error", {
                    "message": (
                        f"抱歉，尝试 {MAX_RETRIES + 1} 次后仍无法完成请求。"
                        f"原因：{error_text}"
                    )
                })
                failure = a2ui_payload_if_enabled(
                    cfg,
                    {"type": "invoke_failed", "error": error_text},
                )
                if failure is not None:
                    yield encode_sse("a2ui", failure)
                yield encode_sse("done", {"ok": False})
                return

            try:
                metric_id = session.context.persist_layer_metrics(result)
                if metric_id:
                    logger.info(
                        "[session=%s] persisted layer metrics: %s",
                        session_id,
                        metric_id,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[session=%s] failed to persist layer metrics: %s",
                    session_id,
                    exc,
                )

            feedback = session.context.build_quality_feedback(result)
            if feedback.get("enabled"):
                feedback_payload = a2ui_payload_if_enabled(cfg, {
                    "type": "quality_feedback",
                    "status": feedback.get("status"),
                    "failure_count": feedback.get("failure_count", 0),
                    "failures": feedback.get("failures", []),
                })
                if feedback_payload is not None:
                    yield encode_sse("a2ui", feedback_payload)

            raw_messages = result.get("messages") or []
            request_info_form = _check_request_info_form(raw_messages)
            if request_info_form is not None:
                form_payload = a2ui_payload_if_enabled(cfg, request_info_form)
                if form_payload is not None:
                    yield encode_sse("a2ui", form_payload)
                hint = "💡 请在上方表单中补充信息，我会根据你的回答继续规划。"
                yield encode_sse("token", {"content": hint})
                if session.messages and session.messages[-1] is user_message:
                    session.messages.pop()
                committed = True
                logger.info(
                    "[session=%s] request_travel_info detected, awaiting form_response",
                    session_id,
                )
                yield encode_sse("done", {"ok": True, "awaiting_form": True})
                return

            final_text = _extract_final_text(raw_messages)
            if not streamed_text_parts:
                yield encode_sse("token", {
                    "content": final_text or "(没有生成回复)",
                })

            call_id_to_name = build_call_id_to_name(raw_messages)
            map_blocks = extract_map_blocks(raw_messages, call_id_to_name)
            weather_block = extract_weather_block(raw_messages)

            for place_card in extract_place_cards(raw_messages, call_id_to_name):
                card_payload = a2ui_payload_if_enabled(cfg, place_card)
                if card_payload is not None:
                    yield encode_sse("a2ui", card_payload)

            if map_blocks:
                yield encode_sse("map_data", {"content": map_blocks})
            if weather_block:
                yield encode_sse("weather_data", {"content": weather_block})

            tool_names = []
            for message in raw_messages:
                for tool_call in getattr(message, "tool_calls", []) or []:
                    if tool_call.get("name"):
                        tool_names.append(tool_call["name"])
            block_types = re.findall(r'"__type":\s*"([^"]+)"', map_blocks)
            logger.info(
                "[session=%s] tools=%s map_blocks=%s weather=%s",
                session_id,
                tool_names,
                block_types,
                bool(weather_block),
            )

            session.messages = clean_messages_for_next_turn(raw_messages)
            committed = True
            yield encode_sse("done", {"ok": True})
    except asyncio.CancelledError:
        raise
    finally:
        if user_message is not None and not committed:
            # 初始化/压缩/客户端断开时，不把没有完整回答的用户消息留在历史里。
            session.messages = [
                message for message in session.messages if message is not user_message
            ]
        session.last_access = time.monotonic()


def _extract_final_text(raw_messages: list) -> Optional[str]:
    for message in reversed(raw_messages):
        if isinstance(message, HumanMessage):
            break
        content = getattr(message, "content", None)
        if content and not isinstance(message, ToolMessage):
            text = normalize_content(content)
            if text:
                return text
    return None


def _check_request_info_form(raw_messages: list) -> Optional[Dict[str, Any]]:
    """检测 LLM 是否调用了 request_travel_info 工具。"""
    for message in raw_messages:
        if not isinstance(message, ToolMessage):
            continue
        tool_call_id = getattr(message, "tool_call_id", "") or ""
        for ai_message in raw_messages:
            if not isinstance(ai_message, AIMessage):
                continue
            for tool_call in getattr(ai_message, "tool_calls", []) or []:
                if (
                    tool_call.get("id") != tool_call_id
                    or tool_call.get("name") != "request_travel_info"
                ):
                    continue
                raw = getattr(message, "content", "") or ""
                if isinstance(raw, list):
                    raw = next(
                        (
                            block["text"]
                            for block in raw
                            if isinstance(block, dict) and block.get("type") == "text"
                        ),
                        "",
                    )
                try:
                    payload = json.loads(raw) if isinstance(raw, str) else raw
                    if isinstance(payload, dict) and "result" in payload:
                        inner = payload["result"]
                        if isinstance(inner, str):
                            inner = json.loads(inner)
                        if isinstance(inner, dict) and inner.get("__a2ui_form"):
                            return {
                                "type": "form_card",
                                "id": inner.get("id", "llm_request_info"),
                                "reason": "llm_determined",
                                "title": inner.get("title", "完善旅行信息"),
                                "message": inner.get("message", ""),
                                "fields": inner.get("fields", []),
                            }
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
    return None

