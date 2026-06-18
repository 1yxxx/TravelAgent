"""
Agent 后端的 A2UI 卡片 Payload 生成器。

当前生成两类卡片：
- ``form_card``：根据前置校验失败原因生成信息补充表单；
- ``place_card``：从搜索类 ToolMessage 中提取景点/酒店/餐厅卡片。

本模块只构造结构化字典，实际 WebSocket 发送由 ``agent_fastapi.py``
负责，浏览器渲染由 ``web/static/a2ui-cards.js`` 负责。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

# ── 不同前置校验失败原因对应的表单字段 ───────────────────────────────────────

_FORM_FIELD_CONFIGS: Dict[str, List[Dict[str, Any]]] = {
    "query_too_short": [
        {
            "key": "destination", "label": "目的地城市", "field_type": "text",
            "placeholder": "例如：成都", "required": True,
        },
        {
            "key": "days", "label": "出行天数", "field_type": "number",
            "placeholder": "例如：3", "required": True, "min": 1, "max": 30,
        },
        {
            "key": "budget", "label": "预算范围", "field_type": "select",
            "options": [
                {"label": "经济实惠（2000以内）", "value": "budget"},
                {"label": "舒适享受（2000-5000）", "value": "mid"},
                {"label": "豪华体验（5000以上）", "value": "luxury"},
            ],
        },
        {
            "key": "preference", "label": "旅行偏好", "field_type": "textarea",
            "placeholder": "例如：亲子游、美食之旅、历史文化、自然风光…",
        },
    ],
    "intent_unclear": [
        {
            "key": "destination", "label": "目的地城市", "field_type": "text",
            "placeholder": "例如：成都", "required": True,
        },
        {
            "key": "days", "label": "出行天数", "field_type": "number",
            "placeholder": "例如：3", "required": True, "min": 1, "max": 30,
        },
        {
            "key": "preference", "label": "旅行偏好", "field_type": "textarea",
            "placeholder": "例如：亲子游、美食之旅…",
        },
    ],
    "missing_destination": [
        {
            "key": "destination", "label": "目的地城市", "field_type": "text",
            "placeholder": "例如：杭州、成都、青岛", "required": True,
        },
        {
            "key": "days", "label": "出行天数", "field_type": "number",
            "placeholder": "例如：3", "required": True, "min": 1, "max": 30,
        },
        {
            "key": "budget", "label": "预算范围", "field_type": "select",
            "options": [
                {"label": "经济实惠", "value": "budget"},
                {"label": "舒适享受", "value": "mid"},
                {"label": "豪华体验", "value": "luxury"},
            ],
        },
        {
            "key": "preference", "label": "旅行偏好", "field_type": "textarea",
            "placeholder": "例如：亲子游、美食之旅…",
        },
    ],
    "missing_duration": [
        {
            "key": "days", "label": "出行天数", "field_type": "number",
            "placeholder": "例如：3", "required": True, "min": 1, "max": 30,
        },
        {
            "key": "budget", "label": "预算范围", "field_type": "select",
            "options": [
                {"label": "经济实惠", "value": "budget"},
                {"label": "舒适享受", "value": "mid"},
                {"label": "豪华体验", "value": "luxury"},
            ],
        },
        {
            "key": "preference", "label": "旅行偏好", "field_type": "textarea",
            "placeholder": "例如：亲子游、美食之旅…",
        },
    ],
}

_FORM_TITLES: Dict[str, str] = {
    "query_too_short":     "完善旅行信息",
    "intent_unclear":      "旅程规划表",
    "missing_destination": "目的地信息",
    "missing_duration":    "出行天数",
}


def build_form_card_payload(precheck: Dict[str, Any]) -> Dict[str, Any]:
    """把失败的前置校验结果转换为 A2UI ``form_card``。

    参数：
        precheck: 包含 ``ok=False``、``reason`` 和 ``suggestion`` 的字典。

    返回：
        可直接交给 ``_send_a2ui_if_enabled`` 的 A2UI Payload。
    """
    reason = precheck.get("reason", "missing_destination")
    fields = _FORM_FIELD_CONFIGS.get(reason, _FORM_FIELD_CONFIGS["missing_destination"])
    title = _FORM_TITLES.get(reason, "完善旅行信息")

    return {
        "type": "form_card",
        "id": f"precheck_{reason}",
        "reason": reason,
        "title": title,
        "message": precheck.get("suggestion", "请补充以下信息以便为你规划："),
        "fields": fields,
    }


# ── 从 ToolMessage 提取地点卡片 ──────────────────────────────────────────────

_TOOL_CATEGORY_MAP: Dict[str, str] = {
    "search_poi":        "poi",
    "search_hotel":      "hotel",
    "search_restaurant": "restaurant",
}

_CATEGORY_LABELS: Dict[str, str] = {
    "poi":        "🏛️ 景点",
    "hotel":      "🏨 酒店",
    "restaurant": "🍜 餐厅",
}


def extract_place_cards(
    messages: list,
    call_id_to_name: Dict[str, str],
    max_places: int = 3,
) -> List[Dict[str, Any]]:
    """扫描搜索工具产生的 ToolMessage，生成 A2UI ``place_card``。

    同一轮中，同一种搜索工具最多生成一张卡片，避免 Agent 重复调用工具时
    前端出现大量重复内容。

    参数：
        messages: 当前轮 Agent 返回的完整消息列表；
        call_id_to_name: ``tool_call_id -> 工具名`` 映射；
        max_places: 每张卡片最多展示的地点数量。

    返回：
        A2UI 地点卡片 Payload 列表。
    """
    from langchain_core.messages import ToolMessage

    cards: List[Dict[str, Any]] = []
    seen_tool_names: set[str] = set()

    for m in messages:
        if not isinstance(m, ToolMessage):
            continue

        tname = call_id_to_name.get(getattr(m, "tool_call_id", "") or "", "")
        if tname not in _TOOL_CATEGORY_MAP:
            continue
        # 同一轮内每类搜索工具只生成一张卡片，避免重复渲染。
        if tname in seen_tool_names:
            continue
        seen_tool_names.add(tname)

        raw = getattr(m, "content", "") or ""
        # MCP Adapter 可能把文本包装为 content block 列表，先还原原始文本。
        if isinstance(raw, list):
            raw = next((b["text"] for b in raw if isinstance(b, dict) and b.get("type") == "text"), "")

        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue

        # 解开 MCP 统一信封 {artifact_id, result, isError}。
        if isinstance(payload, dict) and "result" in payload and "artifact_id" in payload:
            if payload.get("isError"):
                continue
            payload = payload["result"]
        # 部分工具会在 result 中再次返回 JSON 字符串，需要二次反序列化。
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue

        items = payload if isinstance(payload, list) else []
        if not items:
            continue

        category = _TOOL_CATEGORY_MAP[tname]
        category_label = _CATEGORY_LABELS.get(category, "📍 地点")
        city = (items[0].get("cityname") or "") if items else ""

        places: List[Dict[str, Any]] = []
        for item in items[:max_places]:
            if not isinstance(item, dict):
                continue

            # 高德照片字段可能是列表，也可能包在 {"photo": [...]} 中。
            raw_photos = item.get("photos") or []
            if isinstance(raw_photos, dict):
                raw_photos = raw_photos.get("photo") or []
            photos: List[str] = []
            for ph in raw_photos[:3]:
                url = ph.get("url") if isinstance(ph, dict) else str(ph)
                if url and url.startswith("http"):
                    photos.append(url)

            # 前端地图只接受有效数值坐标；解析失败时显式返回 None。
            lng = None
            lat = None
            try:
                lng = float(item.get("longitude") or 0)
                lat = float(item.get("latitude") or 0)
            except (TypeError, ValueError):
                pass
            if not lng or not lat:
                lng = None
                lat = None

            # 将评分和人均消费压缩成卡片摘要，避免前端重复拼装业务格式。
            meta_parts: List[str] = []
            if item.get("rating"):
                meta_parts.append(f"⭐{item['rating']}")
            if item.get("cost"):
                meta_parts.append(f"¥{item['cost']}")
            summary = " · ".join(meta_parts) if meta_parts else ""

            places.append({
                "id":        str(item.get("id") or item.get("name", "")),
                "name":      str(item.get("name") or "地点"),
                "category":  category,
                "label":     category_label,
                "address":   str(item.get("address") or ""),
                "tel":       str(item.get("tel") or ""),
                "rating":    str(item.get("rating") or ""),
                "cost":      str(item.get("cost") or ""),
                "cuisine":   str(item.get("cuisine") or ""),
                "photos":    photos,
                "longitude": lng,
                "latitude":  lat,
                "summary":   summary,
            })

        if places:
            cards.append({
                "type":           "place_card",
                "source_tool":    tname,
                "category":       category,
                "category_label": category_label,
                "city":           city,
                "places":         places,
            })

    return cards
