"""
A2UI card payload generation for the travel agent.

Generates two card types:
- form_card: built from ClientContext.precheck_user_request() failure results
- place_card: extracted from search tool execution results in agent messages
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

# ── Form field configurations per precheck reason ──────────────────────────────

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
    """Build a form_card A2UI payload from a failed precheck result.

    Args:
        precheck: dict with ``ok=False``, ``reason``, and ``suggestion``.

    Returns:
        A2UI payload dict ready for ``_send_a2ui_if_enabled``.
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


# ── Place card extraction ──────────────────────────────────────────────────────

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
    """Scan ToolMessages from search tools and generate place_card A2UI payloads.

    Each distinct search tool call produces one place_card event.

    Args:
        messages: the agent's full message list from the current turn.
        call_id_to_name: mapping from tool_call_id → tool name.
        max_places: max POI items per card.

    Returns:
        List of A2UI place_card payload dicts.
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
        # One card per distinct tool name per turn
        if tname in seen_tool_names:
            continue
        seen_tool_names.add(tname)

        raw = getattr(m, "content", "") or ""
        # langchain-mcp-adapters wraps content as [{"type":"text","text":"..."}]
        if isinstance(raw, list):
            raw = next((b["text"] for b in raw if isinstance(b, dict) and b.get("type") == "text"), "")

        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue

        # Unwrap MCP envelope {artifact_id, result, isError}
        if isinstance(payload, dict) and "result" in payload and "artifact_id" in payload:
            if payload.get("isError"):
                continue
            payload = payload["result"]
        # Some render-like tools return JSON strings
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

            # Extract photos
            raw_photos = item.get("photos") or []
            if isinstance(raw_photos, dict):
                raw_photos = raw_photos.get("photo") or []
            photos: List[str] = []
            for ph in raw_photos[:3]:
                url = ph.get("url") if isinstance(ph, dict) else str(ph)
                if url and url.startswith("http"):
                    photos.append(url)

            # Parse coordinates
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

            # Build summary line
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
