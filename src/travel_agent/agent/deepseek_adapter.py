"""
DeepSeek API 协议适配层。

解决 DeepSeek API 不接受 list 类型 content 的问题，同时回注 reasoning_content。
"""

from __future__ import annotations

import json as _json
from typing import List

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI


def _flatten_content(content) -> str:
    """把 list/dict 类型的 message content 拍平为字符串，DeepSeek 只接受 string。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or _json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return _json.dumps(content, ensure_ascii=False)


def _estimate_tokens(messages: List[BaseMessage]) -> int:
    """粗略估算 token，用于上下文爆炸保护。"""
    total = 0
    for msg in messages:
        content = _flatten_content(getattr(msg, "content", "") or "")
        for ch in content:
            total += 2 if "\u4e00" <= ch <= "\u9fff" else 1
    return total // 4


class DeepSeekChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI 的子类，在发送请求前将所有消息的 content 强制转为 string，
    解决 DeepSeek API 不接受 list 类型 content 的问题。
    同时回注 reasoning_content（DeepSeek thinking 模式要求回传）。
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # 从原始消息中收集 reasoning_content
        reasoning_list: list[str] = []
        if isinstance(input_, list):
            for m in input_:
                rc = getattr(m, "reasoning_content", None)
                if rc:
                    reasoning_list.append(rc)

        fixed = []
        rc_idx = 0
        for msg in payload.get("messages", []):
            content = msg.get("content")
            if content is not None and not isinstance(content, str):
                msg = dict(msg)
                msg["content"] = _flatten_content(content)
            # 将 reasoning_content 回注到对应的 assistant 消息
            if msg.get("role") == "assistant" and rc_idx < len(reasoning_list):
                if not isinstance(msg, dict):
                    msg = dict(msg)
                msg["reasoning_content"] = reasoning_list[rc_idx]
                rc_idx += 1
            fixed.append(msg)
        payload["messages"] = fixed
        return payload
