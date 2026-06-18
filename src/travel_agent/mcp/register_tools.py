"""
MCP 工具注册（重构版）。

使用 ``adapters.py`` 中的共享辅助函数，将 17 个旅行工具注册到 FastMCP Server。
每个工具仍保留显式参数注解（MCP JSON Schema 生成所需），
但工具体通过 ``_execute_tool()`` / ``_execute_tool_sync()`` 统一处理，
消除了原先 25KB 中 ~80% 的重复模板代码。
"""

from __future__ import annotations

from typing import Annotated, Optional

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from pydantic import Field

from travel_agent.config import Settings
from travel_agent.mcp.adapters import (
    _get_session_id, _get_store, _import_tool,
)
from travel_agent.storage.agent_memory import ArtifactStore

try:
    from travel_agent.utils.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)


async def _execute_tool(
    tool_name: str,
    ctx: Context[ServerSession, object],
    cfg: Settings,
    invoke_args: dict,
    *,
    persist: bool = True,
) -> dict:
    """通用异步工具执行体：导入 → 调用 → 持久化 → 返回 MCP 信封。"""
    store = _get_store(ctx, cfg)
    try:
        core_tool = _import_tool(tool_name)
        if core_tool is None:
            raise RuntimeError(f"Unknown tool: {tool_name}")
        result = await core_tool.ainvoke(invoke_args)
        if persist:
            meta = store.save_result(node_id=tool_name, payload=result, summary=f"{tool_name} result")
            return {"artifact_id": meta.artifact_id, "result": result, "isError": False}
        return {"artifact_id": "", "result": result, "isError": False}
    except Exception as exc:
        import traceback
        logger.error("[MCP %s] %s\n%s", tool_name, exc, traceback.format_exc())
        return {"artifact_id": "", "result": str(exc), "isError": True}


def _execute_tool_sync(
    tool_name: str,
    ctx: Context[ServerSession, object],
    cfg: Settings,
    invoke_args: dict,
    *,
    persist: bool = True,
) -> dict:
    """通用同步工具执行体。"""
    store = _get_store(ctx, cfg)
    try:
        core_tool = _import_tool(tool_name)
        if core_tool is None:
            raise RuntimeError(f"Unknown tool: {tool_name}")
        result = core_tool.invoke(invoke_args)
        if persist:
            meta = store.save_result(node_id=tool_name, payload=result, summary=f"{tool_name} result")
            return {"artifact_id": meta.artifact_id, "result": result, "isError": False}
        return {"artifact_id": "", "result": result, "isError": False}
    except Exception as exc:
        import traceback
        logger.error("[MCP %s] %s\n%s", tool_name, exc, traceback.format_exc())
        return {"artifact_id": "", "result": str(exc), "isError": True}


def register(server: FastMCP, cfg: Settings) -> None:
    """把全部旅行工具注册到指定 FastMCP Server。"""

    # ── search_poi ──
    @server.tool(name="search_poi", description="搜索城市内的景点、酒店、餐厅等地点（POI），返回名称、地址、经纬度等信息。")
    async def mcp_search_poi(
        ctx: Context[ServerSession, object],
        city: Annotated[str, Field(description="目标城市名称，例如 '北京'")],
        keyword: Annotated[str, Field(description="搜索关键词，例如 '故宫'")],
        types: Annotated[Optional[str], Field(description="POI 类型编码，例如 '110202'（景点）")] = None,
        max_results: Annotated[int, Field(description="返回结果数量，默认 5")] = 5,
    ) -> dict:
        return await _execute_tool("search_poi", ctx, cfg, {"city": city, "keyword": keyword, "types": types, "max_results": max_results})

    # ── check_weather ──
    @server.tool(name="check_weather", description="查询指定城市的实时天气或未来 3 天天气预报。")
    async def mcp_check_weather(
        ctx: Context[ServerSession, object],
        city: Annotated[str, Field(description="城市名称，例如 '上海'")],
        forecast: Annotated[bool, Field(description="True 返回 3 天预报，False 返回实时天气")] = True,
    ) -> dict:
        return await _execute_tool("check_weather", ctx, cfg, {"city": city, "forecast": forecast})

    # ── search_hotel ──
    @server.tool(name="search_hotel", description="搜索指定城市的酒店，支持按价格档次和关键词筛选。")
    async def mcp_search_hotel(
        ctx: Context[ServerSession, object],
        city: Annotated[str, Field(description="城市名称")],
        keyword: Annotated[str, Field(description="关键词，例如 '五星级' 或 '快捷酒店'")] = "",
        budget_level: Annotated[str, Field(description="价格档次: economy/mid/luxury")] = "mid",
        max_results: Annotated[int, Field(description="返回结果数量，默认 5")] = 5,
    ) -> dict:
        return await _execute_tool("search_hotel", ctx, cfg, {"city": city, "keyword": keyword, "budget_level": budget_level, "max_results": max_results})

    # ── search_restaurant ──
    @server.tool(name="search_restaurant", description="搜索指定城市的餐厅，支持按菜系和关键词筛选。")
    async def mcp_search_restaurant(
        ctx: Context[ServerSession, object],
        city: Annotated[str, Field(description="城市名称")],
        keyword: Annotated[str, Field(description="关键词，例如 '川菜' 或 '火锅'")] = "",
        max_results: Annotated[int, Field(description="返回结果数量，默认 5")] = 5,
    ) -> dict:
        return await _execute_tool("search_restaurant", ctx, cfg, {"city": city, "keyword": keyword, "max_results": max_results})

    # ── plan_route ──
    @server.tool(name="plan_route", description="规划两点之间的驾车路线，返回距离、时长和折线坐标。")
    async def mcp_plan_route(
        ctx: Context[ServerSession, object],
        origin: Annotated[str, Field(description="出发地名称或经纬度 '116.4,39.9'")],
        destination: Annotated[str, Field(description="目的地名称或经纬度")],
        city: Annotated[str, Field(description="所在城市，用于地名解析")] = "",
    ) -> dict:
        return await _execute_tool("plan_route", ctx, cfg, {"origin": origin, "destination": destination, "city": city})

    # ── read_artifact ──
    @server.tool(name="read_artifact", description="通过 artifact_id 读取之前工具调用的持久化结果。")
    async def mcp_read_artifact(
        ctx: Context[ServerSession, object],
        artifact_id: Annotated[str, Field(description="要读取的 artifact_id")],
    ) -> dict:
        store = _get_store(ctx, cfg)
        meta, data = store.load_result(artifact_id)
        if meta is None:
            return {"artifact_id": artifact_id, "result": data, "isError": True}
        return {"artifact_id": artifact_id, "result": data, "isError": False}

    # ── plan_itinerary ──
    @server.tool(name="plan_itinerary", description="基于 POI 列表生成结构化多日行程草案（简单均分版，精细规划请用 smart_plan_itinerary）。")
    async def mcp_plan_itinerary(
        ctx: Context[ServerSession, object],
        city: Annotated[str, Field(description="目标城市名称")],
        days: Annotated[int, Field(description="行程天数")],
        pois: Annotated[list, Field(description="POI 列表，每项含 name/longitude/latitude 等字段")],
        preference: Annotated[str, Field(description="游览偏好描述，例如 '历史文化'")] = "",
    ) -> dict:
        return _execute_tool_sync("plan_itinerary", ctx, cfg, {"city": city, "days": days, "pois": pois, "preference": preference})

    # ── smart_plan_itinerary ──
    @server.tool(name="smart_plan_itinerary", description="智能行程分组：K-means 地理聚类 + 节奏控制，把景点/酒店/餐厅科学分配到各天。")
    async def mcp_smart_plan_itinerary(
        ctx: Context[ServerSession, object],
        spots: Annotated[list, Field(description="景点列表，每项含 name/longitude/latitude")],
        hotels: Annotated[list, Field(description="酒店列表")],
        restaurants: Annotated[list, Field(description="餐厅列表")],
        days: Annotated[int, Field(description="总天数")],
        city: Annotated[str, Field(description="城市名称")] = "",
        title: Annotated[str, Field(description="行程标题")] = "",
        weather_summary: Annotated[str, Field(description="天气概况，用于识别雨天")] = "",
        pace: Annotated[str, Field(description="节奏: relaxed/standard/intensive")] = "standard",
    ) -> dict:
        return _execute_tool_sync("smart_plan_itinerary", ctx, cfg, {
            "spots": spots, "hotels": hotels, "restaurants": restaurants,
            "days": days, "city": city, "title": title,
            "weather_summary": weather_summary, "pace": pace,
        })

    # ── estimate_budget ──
    @server.tool(name="estimate_budget", description="粗略估算旅行预算，返回每日花费和总花费估算。")
    async def mcp_estimate_budget(
        ctx: Context[ServerSession, object],
        days: Annotated[int, Field(description="行程天数")],
        city_level: Annotated[str, Field(description="城市等级: A（一线）/B（二线）/C（三线）")] = "A",
        hotel_level: Annotated[str, Field(description="酒店档次: budget/mid/luxury")] = "mid",
        with_flight: Annotated[bool, Field(description="是否含往返机票")] = True,
    ) -> dict:
        return _execute_tool_sync("estimate_budget", ctx, cfg, {"days": days, "city_level": city_level, "hotel_level": hotel_level, "with_flight": with_flight})

    # ── recommend_transport ──
    @server.tool(name="recommend_transport", description="根据距离和城市给出交通方式建议（步行/骑行/地铁/城际）。")
    async def mcp_recommend_transport(
        ctx: Context[ServerSession, object],
        distance_km: Annotated[float, Field(description="距离（千米）")],
        city: Annotated[str, Field(description="所在城市")],
    ) -> dict:
        return _execute_tool_sync("recommend_transport", ctx, cfg, {"distance_km": distance_km, "city": city})

    # ── format_itinerary ──
    @server.tool(name="format_itinerary", description="将已收集的旅行数据整理成结构化 Markdown 格式行程报告（调用 LLM 生成）。")
    async def mcp_format_itinerary(
        ctx: Context[ServerSession, object],
        city: Annotated[str, Field(description="目标城市名称")],
        days: Annotated[int, Field(description="行程天数")],
        travelers: Annotated[int, Field(description="出行人数")] = 2,
        budget: Annotated[str, Field(description="预算描述，如 '500元/天'")] = "适中",
        raw_data: Annotated[str, Field(description="各工具返回的数据汇总（JSON 字符串或文本）")] = "",
    ) -> dict:
        return await _execute_tool("format_itinerary", ctx, cfg, {"city": city, "days": days, "travelers": travelers, "budget": budget, "raw_data": raw_data})

    # ── render_map_pois ──
    @server.tool(name="render_map_pois", description="将 POI 列表打包为前端地图可渲染的标记数据（不调用外部 API）。")
    async def mcp_render_map_pois(
        ctx: Context[ServerSession, object],
        items: Annotated[list, Field(description="POI 列表，每项含 name/longitude/latitude/type")],
        title: Annotated[Optional[str], Field(description="标注组标题")] = None,
    ) -> dict:
        return _execute_tool_sync("render_map_pois", ctx, cfg, {"items": items, "title": title})

    # ── render_map_route ──
    @server.tool(name="render_map_route", description="将路线折线坐标打包为前端地图可渲染的路线数据。")
    async def mcp_render_map_route(
        ctx: Context[ServerSession, object],
        polyline: Annotated[list, Field(description="折线坐标列表，每项为 [lng, lat]")],
        origin: Annotated[str, Field(description="出发地名称")] = "",
        destination: Annotated[str, Field(description="目的地名称")] = "",
        distance_km: Annotated[Optional[float], Field(description="距离（千米）")] = None,
        duration_min: Annotated[Optional[float], Field(description="时长（分钟）")] = None,
    ) -> dict:
        return _execute_tool_sync("render_map_route", ctx, cfg, {"polyline": polyline, "origin": origin, "destination": destination, "distance_km": distance_km, "duration_min": duration_min})

    # ── render_itinerary ──
    @server.tool(name="render_itinerary", description="将完整行程规划打包为地图可渲染的有序景点数据，前端按天分组标注并连线。")
    async def mcp_render_itinerary(
        ctx: Context[ServerSession, object],
        days: Annotated[list, Field(description="每天安排列表，每项含 day/label/spots")],
        city: Annotated[str, Field(description="城市名称")] = "",
        title: Annotated[str, Field(description="行程标题")] = "",
    ) -> dict:
        return _execute_tool_sync("render_itinerary", ctx, cfg, {"days": days, "city": city, "title": title})

    # ── validate_json ──
    @server.tool(name="validate_json", description="检查字符串是否为合法 JSON。")
    async def mcp_validate_json(
        ctx: Context[ServerSession, object],
        payload: Annotated[str, Field(description="待检查的字符串")],
    ) -> dict:
        return _execute_tool_sync("validate_json", ctx, cfg, {"payload": payload}, persist=False)

    # ── fix_json ──
    @server.tool(name="fix_json", description="利用 LLM 对接近 JSON 但不合法的文本进行纠错。")
    async def mcp_fix_json(
        ctx: Context[ServerSession, object],
        raw_text: Annotated[str, Field(description="原始字符串（可能含多余说明、尾逗号等）")],
        instruction: Annotated[str, Field(description="可选 schema 提示")] = "",
    ) -> dict:
        return await _execute_tool("fix_json", ctx, cfg, {"raw_text": raw_text, "instruction": instruction}, persist=False)

    # ── request_travel_info ──
    @server.tool(name="request_travel_info", description="当用户的旅行需求信息不完整时调用此工具，弹出表单让用户补充目的地、天数、预算、偏好等信息。只在确实缺少关键信息时调用。")
    async def mcp_request_travel_info(
        ctx: Context[ServerSession, object],
        missing_fields: Annotated[str, Field(description="缺少的字段列表，逗号分隔。可选值：destination, days, budget, preference。例如 'destination,days'")] = "",
        custom_message: Annotated[str, Field(description="友好的提示语，显示在表单上方")] = "",
    ) -> dict:
        return _execute_tool_sync("request_travel_info", ctx, cfg, {"missing_fields": missing_fields, "custom_message": custom_message or None})

    logger.info("[MCP] registered all 17 travel tools")
