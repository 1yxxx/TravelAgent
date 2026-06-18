"""
WebSocket 聊天端点处理模块。

本模块只负责 WebSocket 连接的完整请求-响应流程：
接收消息 → 调用 Agent → 提取地图/天气/卡片 → 推送回复。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from travel_agent.agent.factory import build_agent
from travel_agent.config import load_settings
from travel_agent.api.message_utils import normalize_content, clean_messages_for_next_turn
from travel_agent.api.a2ui_bridge import send_a2ui_if_enabled, extract_place_cards
from travel_agent.api.map_extractor import (
    extract_map_blocks, extract_weather_block, build_call_id_to_name,
)
from travel_agent.utils.logging import logger

# 超时与重试配置
AGENT_TIMEOUT = 120
MAX_RETRIES = 1
AGENT_RECURSION_LIMIT = 20


async def handle_websocket(ws: WebSocket, config_path: str):
    """处理单个 WebSocket 连接的生命周期。

    在 ``server.py`` 的 ``@app.websocket("/ws/chat")`` 中调用。
    """
    await ws.accept()

    session_id = f"travel_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    cfg = load_settings(config_path)

    try:
        agent, context = await build_agent(cfg=cfg, session_id=session_id, lang="zh")
    except Exception as exc:
        logger.exception("build_agent 失败: %s", exc)
        await ws.send_text(f"后端初始化失败：{exc}")
        await ws.close()
        return

    messages = []

    try:
        while True:
            data = await ws.receive_text()
            if not data:
                continue
            if data.strip() in ("/exit", "/quit"):
                await ws.close()
                break

            # ── 处理前端发来的 A2UI 消息（表单提交等）──
            if data.startswith(cfg.a2ui.event_prefix):
                try:
                    a2ui_payload = json.loads(data[len(cfg.a2ui.event_prefix):])
                except json.JSONDecodeError:
                    continue
                if a2ui_payload.get("type") == "form_response":
                    fd = a2ui_payload.get("data") or {}
                    parts = []
                    if fd.get("destination"):
                        parts.append(f"目的地：{fd['destination']}")
                    if fd.get("days"):
                        parts.append(f"天数：{fd['days']}天")
                    if fd.get("budget"):
                        budget_labels = {
                            "budget": "经济实惠", "mid": "舒适享受", "luxury": "豪华体验"
                        }
                        parts.append(f"预算：{budget_labels.get(fd['budget'], fd['budget'])}")
                    if fd.get("preference"):
                        parts.append(f"偏好：{fd['preference']}")
                    if parts:
                        data = "，".join(parts) + "。请帮我规划旅行。"
                        logger.info("[A2UI] form_response synthesized: %s", data)
                    else:
                        continue
                else:
                    continue

            messages.append(HumanMessage(content=data))

            # ── 记忆框架动态切换 ──
            invoke_messages, memory_meta = await context.prepare_messages_for_invoke(messages)
            await send_a2ui_if_enabled(ws, cfg, {
                "type": "memory_mode",
                "mode": memory_meta.get("mode"),
                "reason": memory_meta.get("reason"),
                "message_count": memory_meta.get("message_count"),
                "token_estimate": memory_meta.get("token_estimate"),
            })

            # ── 超时 + 重试 ──
            result: Optional[Dict[str, Any]] = None
            last_exc: Optional[Exception] = None

            for attempt in range(1, MAX_RETRIES + 2):
                try:
                    result = await asyncio.wait_for(
                        agent.ainvoke(
                            {"messages": invoke_messages},
                            config={"recursion_limit": AGENT_RECURSION_LIMIT},
                        ),
                        timeout=AGENT_TIMEOUT,
                    )
                    last_exc = None
                    break
                except asyncio.TimeoutError:
                    last_exc = asyncio.TimeoutError(f"请求超时（>{AGENT_TIMEOUT}s）")
                    logger.warning("[session=%s] Agent 超时，第 %d 次尝试", session_id, attempt)
                    if attempt <= MAX_RETRIES:
                        await send_a2ui_if_enabled(
                            ws, cfg, {"type": "retry", "attempt": attempt, "reason": "timeout"}
                        )
                        await ws.send_text(f"❗ 请求超时，正在进行第 {attempt} 次重试…")
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.exception("[session=%s] Agent 调用失败（第 %d 次）: %s", session_id, attempt, exc)
                    if attempt <= MAX_RETRIES:
                        await send_a2ui_if_enabled(
                            ws, cfg, {"type": "retry", "attempt": attempt, "reason": str(exc)}
                        )
                        await ws.send_text(f"❗ 发生错误，正在进行第 {attempt} 次重试…")

            if result is None:
                err_msg = str(last_exc) if last_exc else "未知错误"
                await ws.send_text(
                    f"❌ 抱歉，尝试 {MAX_RETRIES + 1} 次后仍无法完成请求。\n原因：{err_msg}"
                )
                await send_a2ui_if_enabled(
                    ws, cfg, {"type": "invoke_failed", "error": err_msg}
                )
                messages = messages[:-1]
                continue

            # ── 分层指标持久化 ──
            try:
                metric_artifact_id = context.persist_layer_metrics(result)
                if metric_artifact_id:
                    logger.info("[session=%s] persisted layer metrics: %s", session_id, metric_artifact_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[session=%s] failed to persist layer metrics: %s", session_id, exc)

            # ── 质量反馈 ──
            feedback = context.build_quality_feedback(result)
            if feedback.get("enabled"):
                await send_a2ui_if_enabled(ws, cfg, {
                    "type": "quality_feedback",
                    "status": feedback.get("status"),
                    "failure_count": feedback.get("failure_count", 0),
                    "failures": feedback.get("failures", []),
                })

            raw_messages = result["messages"]

            # ── 检测 request_travel_info 表单 ──
            _request_info_form = _check_request_info_form(raw_messages)
            if _request_info_form is not None:
                await send_a2ui_if_enabled(ws, cfg, _request_info_form)
                await ws.send_text("💡 请在上方表单中补充信息，我会根据你的回答继续规划。")
                messages.pop()
                logger.info("[session=%s] request_travel_info detected, awaiting form_response", session_id)
                continue

            # ── 提取最终回复文本 ──
            final_text = None
            for m in reversed(raw_messages):
                if isinstance(m, HumanMessage):
                    break
                content = getattr(m, "content", None)
                if content and not isinstance(m, ToolMessage):
                    final_text = normalize_content(content)
                    if final_text:
                        break

            # ── 构建 tool_call_id → tool_name 索引 ──
            call_id_to_name = build_call_id_to_name(raw_messages)

            # ── 追加地图 / 天气 JSON 块 ──
            map_blocks = extract_map_blocks(raw_messages, call_id_to_name)
            weather_block = extract_weather_block(raw_messages)

            # ── 发送 A2UI 地点卡片 ──
            place_cards = extract_place_cards(raw_messages, call_id_to_name)
            for pc in place_cards:
                await send_a2ui_if_enabled(ws, cfg, pc)

            # 调试日志
            _tool_names = []
            for _m in raw_messages:
                for _tc in getattr(_m, "tool_calls", []) or []:
                    if _tc.get("name"):
                        _tool_names.append(_tc["name"])
            _block_types = re.findall(r'"__type":\s*"([^"]+)"', map_blocks)
            logger.info(
                "[session=%s] tools=%s  map_blocks=%s  weather=%s",
                session_id, _tool_names, _block_types, bool(weather_block),
            )

            # 清理消息历史
            messages = clean_messages_for_next_turn(raw_messages)

            reply = final_text or "(没有生成回复)"
            if map_blocks:
                reply = reply + "\n" + map_blocks
            if weather_block:
                reply = reply + "\n" + weather_block

            await ws.send_text(reply)

    except WebSocketDisconnect:
        logger.info("WebSocket 连接断开：%s", session_id)


def _check_request_info_form(raw_messages: list) -> Optional[Dict[str, Any]]:
    """检测 LLM 是否调用了 request_travel_info 工具。

    Returns:
        form_card payload 或 None。
    """
    for m in raw_messages:
        if not isinstance(m, ToolMessage):
            continue
        _tcid = getattr(m, "tool_call_id", "") or ""
        for am in raw_messages:
            if not isinstance(am, AIMessage):
                continue
            for tc in getattr(am, "tool_calls", []) or []:
                if tc.get("id") == _tcid and tc.get("name") == "request_travel_info":
                    _raw = getattr(m, "content", "") or ""
                    if isinstance(_raw, list):
                        _raw = next(
                            (b["text"] for b in _raw if isinstance(b, dict) and b.get("type") == "text"), ""
                        )
                    try:
                        _payload = json.loads(_raw) if isinstance(_raw, str) else _raw
                        if isinstance(_payload, dict) and "result" in _payload:
                            _inner = _payload["result"]
                            if isinstance(_inner, str):
                                _inner = json.loads(_inner)
                            if isinstance(_inner, dict) and _inner.get("__a2ui_form"):
                                return {
                                    "type": "form_card",
                                    "id": _inner.get("id", "llm_request_info"),
                                    "reason": "llm_determined",
                                    "title": _inner.get("title", "完善旅行信息"),
                                    "message": _inner.get("message", ""),
                                    "fields": _inner.get("fields", []),
                                }
                    except Exception:
                        pass
    return None
