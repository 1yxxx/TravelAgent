"""
A2UI 双向卡片协议桥接层。

集中管理与传输协议无关的 A2UI 相关逻辑：
- 根据配置决定是否构造 A2UI payload
- form_card（信息补充表单）生成
- place_card（地点详情卡片）自动提取
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from travel_agent.api.message_utils import extract_mcp_text, unwrap_mcp_envelope, safe_json_load


# ── form_card 表单字段配置 ────────────────────────────────────────────────────

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

# ── place_card 工具映射 ───────────────────────────────────────────────────────

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


# ── A2UI payload 构造 ─────────────────────────────────────────────────────────

def a2ui_payload_if_enabled(
    cfg: Any,
    payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """A2UI 开启时返回 payload，否则返回 ``None``。

    本函数不再关心 WebSocket/SSE 等传输方式，具体编码由 API 层负责。
    """
    if not getattr(cfg, "a2ui", None) or not cfg.a2ui.enabled:
        return None
    return payload


# ── form_card 生成 ─────────────────────────────────────────────────────────────

def build_form_card_payload(precheck: Dict[str, Any]) -> Dict[str, Any]:
    """把失败的前置校验结果转换为 A2UI ``form_card``。

    Args:
        precheck: 包含 ``ok=False``、``reason`` 和 ``suggestion`` 的字典。

    Returns:
        A2UI Payload 字典。
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


# ── place_card 提取 ────────────────────────────────────────────────────────────

def extract_place_cards(
    messages: list,
    call_id_to_name: Dict[str, str],
    max_places: int = 3,
) -> List[Dict[str, Any]]:
    """扫描搜索工具产生的 ToolMessage，生成 A2UI ``place_card``。

    同一轮中，同一种搜索工具最多生成一张卡片。

    Args:
        messages: 当前轮 Agent 返回的完整消息列表。
        call_id_to_name: ``tool_call_id -> 工具名`` 映射。
        max_places: 每张卡片最多展示的地点数量。

    Returns:
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
        if tname in seen_tool_names:
            continue
        seen_tool_names.add(tname)

        raw = getattr(m, "content", "") or ""
        raw = extract_mcp_text(raw)

        payload = safe_json_load(raw)
        if payload is None:
            continue

        payload = unwrap_mcp_envelope(payload)
        if isinstance(payload, str):
            payload = safe_json_load(payload)
            if payload is None:
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

            raw_photos = item.get("photos") or []
            if isinstance(raw_photos, dict):
                raw_photos = raw_photos.get("photo") or []
            photos: List[str] = []
            for ph in raw_photos[:3]:
                url = ph.get("url") if isinstance(ph, dict) else str(ph)
                if url and url.startswith("http"):
                    photos.append(url)

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
