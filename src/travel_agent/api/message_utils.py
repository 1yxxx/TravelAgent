"""
消息序列化与清理工具。

处理 LangChain 消息的 content 拍平（DeepSeek 兼容）和历史清理。
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.messages import (
    AIMessage, ToolMessage, HumanMessage, SystemMessage,
)


def normalize_content(content) -> str:
    """将 list/dict 类型的 message content 序列化为字符串（DeepSeek 只接受 string）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def clean_messages_for_next_turn(messages: list) -> list:
    """
    保留完整消息历史（Human / AI / Tool），仅将 content 中的 list/dict
    序列化为字符串，避免 DeepSeek 400 错误，同时不破坏上下文记忆。
    动态 SystemMessage 每轮重建，不写回历史。
    """
    cleaned = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        content = normalize_content(getattr(m, "content", "") or "")
        if isinstance(m, HumanMessage):
            cleaned.append(HumanMessage(content=content))
        elif isinstance(m, AIMessage):
            extra: dict = {}
            if getattr(m, "tool_calls", None):
                extra["tool_calls"] = m.tool_calls
            if getattr(m, "additional_kwargs", None):
                extra["additional_kwargs"] = m.additional_kwargs
            # DeepSeek thinking 模式要求 reasoning_content 必须回传
            reasoning = getattr(m, "reasoning_content", None)
            if reasoning:
                extra["reasoning_content"] = reasoning
            cleaned.append(AIMessage(content=content, **extra))
        elif isinstance(m, ToolMessage):
            cleaned.append(ToolMessage(content=content, tool_call_id=m.tool_call_id))
        else:
            cleaned.append(m)
    return cleaned


def extract_mcp_text(raw) -> str:
    """从 langchain-mcp-adapters 的 content blocks 中提取文本。"""
    if isinstance(raw, list):
        return next(
            (b["text"] for b in raw if isinstance(b, dict) and b.get("type") == "text"), ""
        )
    return str(raw) if raw else ""


def unwrap_mcp_envelope(payload, default=None):
    """解包 MCP 统一信封 {artifact_id, result, isError}。

    Returns:
        解包后的 result 数据，或原 payload（若不是信封格式）。
    """
    if isinstance(payload, dict) and "result" in payload and "artifact_id" in payload:
        if payload.get("isError"):
            return default
        return payload["result"]
    return payload


def safe_json_load(raw) -> Optional[dict]:
    """安全地将字符串或已解析数据转换为 dict。"""
    try:
        if isinstance(raw, str):
            return json.loads(raw)
        return raw if isinstance(raw, (dict, list)) else None
    except Exception:
        return None
