"""
前置校验模块：在 Agent 执行前检查用户请求的意图、目的地和天数。
"""

from __future__ import annotations

import re
from typing import Dict, Any, List

from langchain_core.messages import BaseMessage, HumanMessage

from travel_agent.agent.deepseek_adapter import _flatten_content

_COMMON_CITY_HINTS = [
    "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "成都", "重庆", "西安",
    "武汉", "长沙", "青岛", "厦门", "三亚", "昆明", "大理", "丽江", "贵阳",
    "哈尔滨", "长春", "沈阳", "天津", "福州", "南昌", "郑州", "济南",
]


def _has_travel_intent(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in (
        "旅行", "旅游", "行程", "攻略", "出游", "自驾", "自由行", "规划", "路线",
        "景点", "酒店", "机票", "trip", "travel", "itinerary", "route",
    ))


def _has_destination(text: str) -> bool:
    if any(city in text for city in _COMMON_CITY_HINTS):
        return True
    if re.search(r"去[\u4e00-\u9fa5]{2,8}", text):
        return True
    return False


def _has_duration(text: str) -> bool:
    t = text.lower()
    if "周末" in text or "假期" in text:
        return True
    return bool(re.search(r"\d+\s*(天|日|晚)", t))


def precheck_user_request(messages: List[BaseMessage]) -> Dict[str, Any]:
    """前置业务校验：意图、目的地、天数。

    Returns:
        dict with ok (bool), reason (str), suggestion (str).
    """
    humans = [m for m in messages if isinstance(m, HumanMessage)]
    if not humans:
        return {"ok": True}

    text = _flatten_content(getattr(humans[-1], "content", "") or "")
    if len(text.strip()) < 2:
        return {
            "ok": False,
            "reason": "query_too_short",
            "suggestion": "请补充目的地和出行天数，例如：成都3天亲子游，预算5000。",
        }

    if not _has_travel_intent(text):
        return {
            "ok": False,
            "reason": "intent_unclear",
            "suggestion": "我可以帮你规划旅行，请告诉我目的地、天数和预算。",
        }

    if not _has_destination(text):
        return {
            "ok": False,
            "reason": "missing_destination",
            "suggestion": "请先告诉我目的地城市，例如：杭州、成都、青岛。",
        }

    if not _has_duration(text):
        return {
            "ok": False,
            "reason": "missing_duration",
            "suggestion": "请补充出行天数，例如：2天、3天或周末两天。",
        }

    return {"ok": True}
