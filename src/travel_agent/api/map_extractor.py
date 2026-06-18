"""
地图/天气数据提取模块。

从 Agent 的 ToolMessage 输出中自动扫描和提取:
- 地图标记数据 (pois/itinerary/route)
- 天气数据
以供前端渲染交互地图。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, ToolMessage

from travel_agent.api.message_utils import extract_mcp_text, unwrap_mcp_envelope
from travel_agent.utils.logging import logger

# 工具名 → marker type 映射
_TOOL_TYPE_MAP: dict[str, str] = {
    "search_hotel":      "hotel",
    "search_restaurant": "restaurant",
    "search_poi":        "poi",
}


def _amap_type_to_marker_type(amap_type: str) -> str:
    """把高德原始 POI 类型字符串映射为前端 marker type。"""
    if not amap_type:
        return "poi"
    t = amap_type.lower()
    if any(k in t for k in ("住宿", "酒店", "宾馆", "旅馆", "民宿", "hostel", "hotel")):
        return "hotel"
    if any(k in t for k in ("餐饮", "美食", "饭店", "餐厅", "小吃", "food", "restaurant")):
        return "restaurant"
    return "poi"


def extract_weather_block(messages: list) -> Optional[str]:
    """从 ToolMessage 中提取天气数据，返回前端可解析的 JSON 块字符串。"""
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            continue
        raw = getattr(m, "content", "") or ""
        raw = extract_mcp_text(raw)

        payload = None
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue

        payload = unwrap_mcp_envelope(payload)

        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue

        if isinstance(payload, dict) and "days" in payload and "city" in payload:
            days_list = payload.get("days") or []
            is_weather = (
                payload.get("__type") == "weather"
                or (
                    isinstance(days_list, list)
                    and days_list
                    and isinstance(days_list[0], dict)
                    and ("date" in days_list[0] or "weather" in days_list[0])
                    and "spots" not in days_list[0]
                )
            )
            if not is_weather:
                continue
            block = {
                "__type": "weather",
                "city": payload.get("city", ""),
                "days": days_list,
            }
            return "\n```json\n" + json.dumps(block, ensure_ascii=False) + "\n```"
    return None


def extract_map_blocks(
    messages: list,
    call_id_to_name: dict[str, str] | None = None,
) -> str:
    """
    自动扫描本轮所有 ToolMessage，从工具输出中提取地图数据，
    生成前端可解析的 ```json 块。

    行程显示策略：
    - 优先使用 smart_plan_itinerary / render_itinerary 的结构化输出
    - 若两者均未调用，才把 search_* 结果展示为候选 pois
    """
    if call_id_to_name is None:
        call_id_to_name = {}
        for m in messages:
            if not isinstance(m, AIMessage):
                continue
            for tc in getattr(m, "tool_calls", []) or []:
                cid = tc.get("id") or ""
                name = tc.get("name") or ""
                if cid and name:
                    call_id_to_name[cid] = name

    blocks: list[str] = []
    has_itinerary_block: bool = False
    itinerary_days: int = 0
    itinerary_city: str = ""
    _seen_itinerary_keys: set[str] = set()
    _seen_pois_keys: set[str] = set()

    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        raw = getattr(m, "content", "") or ""
        if not raw:
            continue
        raw = extract_mcp_text(raw)

        tool_name = call_id_to_name.get(getattr(m, "tool_call_id", "") or "", "")

        payload = None
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue

        payload = unwrap_mcp_envelope(payload, default=payload)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue

        # ── 已有 __type 标记的工具输出 ──
        if isinstance(payload, dict) and payload.get("__type") in ("pois", "route", "itinerary"):
            block_type = payload.get("__type")
            if block_type == "itinerary":
                ikey = f"{payload.get('city','')}:{len(payload.get('days') or [])}"
                if ikey in _seen_itinerary_keys:
                    blocks = [b for b in blocks if not (
                        '"__type": "itinerary"' in b
                        and f'"city": "{payload.get("city","")}"' in b
                    )]
                _seen_itinerary_keys.add(ikey)
                has_itinerary_block = True
            elif block_type == "pois":
                names_key = str(sorted(i.get("name", "") for i in (payload.get("items") or [])))
                if names_key in _seen_pois_keys:
                    continue
                _seen_pois_keys.add(names_key)
            blocks.append("\n```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```")
            continue

        # ── smart_plan_itinerary → itinerary 块 ──
        if tool_name == "smart_plan_itinerary" and isinstance(payload, dict):
            days_data = payload.get("days")
            if days_data:
                itinerary_block = {
                    "__type": "itinerary",
                    "city": payload.get("city", ""),
                    "title": payload.get("title", ""),
                    "days": days_data,
                }
                ikey = f"{payload.get('city','')}:{len(days_data)}"
                if ikey not in _seen_itinerary_keys:
                    blocks.append(
                        "\n```json\n" + json.dumps(itinerary_block, ensure_ascii=False) + "\n```"
                    )
                    _seen_itinerary_keys.add(ikey)
                    has_itinerary_block = True
            continue

        # ── search_* → pois 块 ──
        if tool_name in _TOOL_TYPE_MAP and isinstance(payload, list) and payload:
            default_type = _TOOL_TYPE_MAP[tool_name]
            items: list[dict] = []
            for item in payload[:5]:
                if not isinstance(item, dict):
                    continue
                try:
                    lng = float(item.get("longitude") or 0)
                    lat = float(item.get("latitude") or 0)
                except (TypeError, ValueError):
                    continue
                if not lng or not lat:
                    continue
                marker_type = _amap_type_to_marker_type(str(item.get("type") or ""))
                if marker_type == "poi" and default_type != "poi":
                    marker_type = default_type
                raw_photos = item.get("photos") or []
                if isinstance(raw_photos, dict):
                    raw_photos = raw_photos.get("photo") or []
                photos: list[str] = []
                for ph in raw_photos[:3]:
                    url = ph.get("url") if isinstance(ph, dict) else str(ph)
                    if url and url.startswith("http"):
                        photos.append(url)
                items.append({
                    "name":      str(item.get("name") or "地点"),
                    "longitude": lng,
                    "latitude":  lat,
                    "type":      marker_type,
                    "address":   str(item.get("address") or ""),
                    "tel":       str(item.get("tel") or ""),
                    "rating":    str(item.get("rating") or ""),
                    "cost":      str(item.get("cost") or ""),
                    "cuisine":   str(item.get("cuisine") or ""),
                    "photos":    photos,
                })
            if items:
                _seen_in_block: set[str] = set()
                deduped_items: list[dict] = []
                for _it in items:
                    _k = _it["name"].strip().lower()
                    if _k not in _seen_in_block:
                        _seen_in_block.add(_k)
                        deduped_items.append(_it)
                blocks.append(
                    "\n```json\n" + json.dumps(
                        {"__type": "pois", "items": deduped_items}, ensure_ascii=False
                    ) + "\n```"
                )
            continue

        # ── plan_route → route 块 ──
        if tool_name == "plan_route" and isinstance(payload, dict):
            polyline = payload.get("polyline")
            if polyline and len(polyline) >= 2:
                blocks.append(
                    "\n```json\n" + json.dumps({
                        "__type":       "route",
                        "polyline":     polyline,
                        "origin":       payload.get("origin", ""),
                        "destination":  payload.get("destination", ""),
                        "distance_km":  payload.get("distance_km"),
                        "duration_min": payload.get("duration_min"),
                    }, ensure_ascii=False) + "\n```"
                )
            continue

        # ── format_itinerary → 记录天数/城市 ──
        if tool_name == "format_itinerary" and isinstance(payload, dict):
            itinerary_days = int(payload.get("days") or 0)
            itinerary_city = str(payload.get("city") or "")
            continue

    # ── 兜底：LLM 做了搜索但没调规划工具时，自动合成 itinerary 块 ──
    if not has_itinerary_block:
        _fallback_spots, _fallback_hotels, _fallback_restaurants = _collect_fallback_pois(
            messages, call_id_to_name
        )
        if _fallback_spots:
            _fallback_city, _fallback_days = _extract_city_days_from_toolcalls(messages)
            try:
                from travel_agent.tools.planning.smart_plan_itinerary import smart_plan_itinerary_tool as _spit
                _days = _fallback_days or max(1, (len(_fallback_spots) + 2) // 3)
                _fb_result = _spit.invoke({
                    "spots": _fallback_spots,
                    "hotels": _fallback_hotels,
                    "restaurants": _fallback_restaurants,
                    "days": _days,
                    "city": _fallback_city,
                    "title": f"{_fallback_city} {_days}日行程" if _fallback_city else "",
                    "pace": "standard",
                })
                if _fb_result.get("days"):
                    blocks.append(
                        "\n```json\n" + json.dumps({
                            "__type": "itinerary",
                            "city": _fb_result.get("city", _fallback_city),
                            "title": _fb_result.get("title", ""),
                            "days": _fb_result["days"],
                        }, ensure_ascii=False) + "\n```"
                    )
                    logger.info(
                        "[_extract_map_blocks] fallback itinerary generated: city=%s days=%d spots=%d",
                        _fallback_city, _days, len(_fallback_spots),
                    )
            except Exception as _fb_exc:
                logger.warning("[_extract_map_blocks] fallback itinerary failed: %s", _fb_exc)

    return "".join(blocks)


def _collect_fallback_pois(messages: list, call_id_to_name: dict) -> tuple:
    """扫描所有 search_* ToolMessage 收集有效 POI。"""
    spots, hotels, restaurants = [], [], []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        raw2 = getattr(m, "content", "") or ""
        raw2 = extract_mcp_text(raw2)
        tname = call_id_to_name.get(getattr(m, "tool_call_id", "") or "", "")
        if tname not in _TOOL_TYPE_MAP:
            continue
        try:
            p2 = json.loads(raw2) if isinstance(raw2, str) else raw2
            p2 = unwrap_mcp_envelope(p2, default=p2)
            if isinstance(p2, str):
                p2 = json.loads(p2)
        except Exception:
            continue
        if not isinstance(p2, list):
            continue
        for item in p2:
            if not isinstance(item, dict):
                continue
            try:
                lng = float(item.get("longitude") or 0)
                lat = float(item.get("latitude") or 0)
            except (TypeError, ValueError):
                continue
            if not lng or not lat:
                continue
            poi = {
                "name": str(item.get("name") or "地点"),
                "longitude": lng,
                "latitude": lat,
                "type": tname.replace("search_", "").replace("poi", "poi"),
                "address": str(item.get("address") or ""),
                "tel": str(item.get("tel") or ""),
                "rating": str(item.get("rating") or ""),
                "cost": str(item.get("cost") or ""),
                "photos": [],
                "note": "",
            }
            if tname == "search_hotel":
                hotels.append(poi)
            elif tname == "search_restaurant":
                restaurants.append(poi)
            else:
                spots.append(poi)
    return spots, hotels, restaurants


def _extract_city_days_from_toolcalls(messages: list) -> tuple:
    """从 AIMessage tool_call args 中提取 city 和 days。"""
    city, days = "", 0
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        for tc in getattr(m, "tool_calls", []) or []:
            args = tc.get("args") or {}
            if args.get("city") and not city:
                city = str(args["city"])
            if args.get("days") and not days:
                try:
                    days = int(args["days"])
                except (TypeError, ValueError):
                    pass
    return city, days


def build_call_id_to_name(messages: list) -> dict[str, str]:
    """构建 tool_call_id → tool_name 索引。"""
    index: dict[str, str] = {}
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        for tc in getattr(m, "tool_calls", []) or []:
            cid = tc.get("id") or ""
            name = tc.get("name") or ""
            if cid and name:
                index[cid] = name
    return index
