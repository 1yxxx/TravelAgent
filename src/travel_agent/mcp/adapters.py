"""
MCP 工具适配器工厂。

提供 ``mcp_tool_adapter`` 工厂函数，为每个核心工具生成统一的 MCP 薄包装器。
消除原先 ``register_tools.py`` 中 15 个几乎完全相同的模板代码（~25KB → ~100行）。

每个适配器统一完成：
1. 从 MCP 请求上下文获取 session_id
2. 获取对应会话的 ArtifactStore
3. 延迟加载并调用底层 Core Tool
4. 持久化结果并返回 MCP 统一信封
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from travel_agent.config import Settings
from travel_agent.storage.agent_memory import ArtifactStore

try:
    from travel_agent.utils.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)


# ── 工具导入路径映射 ──────────────────────────────────────────────────────────
_TOOL_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "search_poi":          ("travel_agent.tools.search.search_poi",           "search_poi_tool"),
    "search_hotel":        ("travel_agent.tools.search.search_hotel",         "search_hotel_tool"),
    "search_restaurant":   ("travel_agent.tools.search.search_restaurant",    "search_restaurant_tool"),
    "check_weather":       ("travel_agent.tools.search.check_weather",        "check_weather_tool"),
    "plan_itinerary":      ("travel_agent.tools.planning.plan_itinerary",     "plan_itinerary_tool"),
    "smart_plan_itinerary":("travel_agent.tools.planning.smart_plan_itinerary","smart_plan_itinerary_tool"),
    "plan_route":          ("travel_agent.tools.planning.plan_route",         "plan_route_tool"),
    "estimate_budget":     ("travel_agent.tools.planning.estimate_budget",    "estimate_budget_tool"),
    "recommend_transport": ("travel_agent.tools.planning.recommend_transport","recommend_transport_tool"),
    "format_itinerary":    ("travel_agent.tools.planning.format_itinerary",   "format_itinerary_tool"),
    "render_map_pois":     ("travel_agent.tools.rendering.render_map",        "render_map_pois_tool"),
    "render_map_route":    ("travel_agent.tools.rendering.render_map",        "render_map_route_tool"),
    "render_itinerary":    ("travel_agent.tools.rendering.render_itinerary",  "render_itinerary_tool"),
    "validate_json":       ("travel_agent.tools.utility.json_tools",          "validate_json_tool"),
    "fix_json":            ("travel_agent.tools.utility.json_tools",          "fix_json_tool"),
    "request_travel_info": ("travel_agent.tools.utility.request_travel_info", "request_travel_info_tool"),
    "read_artifact":       (None, None),  # 特殊处理：直接访问 ArtifactStore
}


def _import_tool(tool_name: str):
    """延迟导入工具函数。使用 _TOOL_IMPORT_MAP 查找模块和函数名。"""
    import importlib
    info = _TOOL_IMPORT_MAP.get(tool_name)
    if not info or not info[0]:
        return None
    module_path, func_name = info
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def _get_session_id(ctx: Context) -> str:
    """从 MCP 请求头提取 session_id。"""
    try:
        return ctx.request_context.request.headers.get("X-Travel-Session-Id", "default")
    except Exception:
        return "default"


def _get_store(ctx: Context, cfg: Settings) -> ArtifactStore:
    """从 lifespan 管理器获取当前 session 对应的 ArtifactStore。"""
    session_id = _get_session_id(ctx)
    mgr = ctx.request_context.lifespan_context
    return mgr.get_store(session_id)


def make_tool_handler(
    tool_name: str,
    description: str,
    invoke_arg_names: list[str],
    *,
    persist_artifact: bool = True,
    sync_invoke: bool = False,
) -> Callable:
    """
    创建 MCP 工具的异步处理函数工厂。

    Args:
        tool_name: 核心工具名称（对应 _TOOL_IMPORT_MAP 中的 key）。
        description: 工具的 MCP 描述字符串。
        invoke_arg_names: 需要从 MCP 参数映射到 core_tool.invoke() 的关键字参数名。
        persist_artifact: 是否将结果持久化到 ArtifactStore。
        sync_invoke: 使用 .invoke()（同步）还是 .ainvoke()（异步）。

    Returns:
        async def handler(mcp_ctx, cfg, **kwargs) → dict MCP 信封
    """

    async def handler(mcp_ctx: Context[ServerSession, object], cfg: Settings, **kwargs) -> dict:
        store = _get_store(mcp_ctx, cfg)

        # ── 特殊工具：read_artifact 直接操作 ArtifactStore ──
        if tool_name == "read_artifact":
            artifact_id = kwargs.get("artifact_id", "")
            meta, data = store.load_result(artifact_id)
            if meta is None:
                return {"artifact_id": artifact_id, "result": data, "isError": True}
            return {"artifact_id": artifact_id, "result": data, "isError": False}

        # ── 通用工具：延迟导入 → 调用 → 持久化 ──
        try:
            core_tool = _import_tool(tool_name)
            if core_tool is None:
                raise RuntimeError(f"Unknown tool: {tool_name}")

            # 构建 invoke 参数
            invoke_args = {k: kwargs[k] for k in invoke_arg_names if k in kwargs}
            if sync_invoke:
                result = core_tool.invoke(invoke_args)
            else:
                result = await core_tool.ainvoke(invoke_args)

            if persist_artifact:
                meta = store.save_result(
                    node_id=tool_name,
                    payload=result,
                    summary=f"{tool_name} result",
                )
                return {"artifact_id": meta.artifact_id, "result": result, "isError": False}
            else:
                return {"artifact_id": "", "result": result, "isError": False}

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("[MCP %s] %s\n%s", tool_name, exc, tb)
            return {"artifact_id": "", "result": str(exc), "isError": True}

    return handler


# ── 工具注册元数据表 ───────────────────────────────────────────────────────────

TOOL_REGISTRY = [
    {
        "name": "search_poi",
        "description": "搜索城市内的景点、酒店、餐厅等地点（POI），返回名称、地址、经纬度等信息。",
        "arg_names": ["city", "keyword", "types", "max_results"],
        "sync": False,
    },
    {
        "name": "check_weather",
        "description": "查询指定城市的实时天气或未来 3 天天气预报。",
        "arg_names": ["city", "forecast"],
        "sync": False,
    },
    {
        "name": "search_hotel",
        "description": "搜索指定城市的酒店，支持按价格档次和关键词筛选。",
        "arg_names": ["city", "keyword", "budget_level", "max_results"],
        "sync": False,
    },
    {
        "name": "search_restaurant",
        "description": "搜索指定城市的餐厅，支持按菜系和关键词筛选。",
        "arg_names": ["city", "keyword", "max_results"],
        "sync": False,
    },
    {
        "name": "plan_route",
        "description": "规划两点之间的驾车路线，返回距离、时长和折线坐标。",
        "arg_names": ["origin", "destination", "city"],
        "sync": False,
    },
    {
        "name": "read_artifact",
        "description": "通过 artifact_id 读取之前工具调用的持久化结果。",
        "arg_names": ["artifact_id"],
        "sync": False,
        "persist": False,
    },
    {
        "name": "plan_itinerary",
        "description": "基于 POI 列表生成结构化多日行程草案（简单均分版，精细规划请用 smart_plan_itinerary）。",
        "arg_names": ["city", "days", "pois", "preference"],
        "sync": True,
    },
    {
        "name": "smart_plan_itinerary",
        "description": "智能行程分组：K-means 地理聚类 + 节奏控制，把景点/酒店/餐厅科学分配到各天。",
        "arg_names": ["spots", "hotels", "restaurants", "days", "city", "title", "weather_summary", "pace"],
        "sync": True,
    },
    {
        "name": "estimate_budget",
        "description": "粗略估算旅行预算，返回每日花费和总花费估算。",
        "arg_names": ["days", "city_level", "hotel_level", "with_flight"],
        "sync": True,
    },
    {
        "name": "recommend_transport",
        "description": "根据距离和城市给出交通方式建议（步行/骑行/地铁/城际）。",
        "arg_names": ["distance_km", "city"],
        "sync": True,
    },
    {
        "name": "format_itinerary",
        "description": "将已收集的旅行数据整理成结构化 Markdown 格式行程报告（调用 LLM 生成）。",
        "arg_names": ["city", "days", "travelers", "budget", "raw_data"],
        "sync": False,
    },
    {
        "name": "render_map_pois",
        "description": "将 POI 列表打包为前端地图可渲染的标记数据（不调用外部 API）。",
        "arg_names": ["items", "title"],
        "sync": True,
    },
    {
        "name": "render_map_route",
        "description": "将路线折线坐标打包为前端地图可渲染的路线数据。",
        "arg_names": ["polyline", "origin", "destination", "distance_km", "duration_min"],
        "sync": True,
    },
    {
        "name": "render_itinerary",
        "description": "将完整行程规划打包为地图可渲染的有序景点数据，前端按天分组标注并连线。",
        "arg_names": ["days", "city", "title"],
        "sync": True,
    },
    {
        "name": "validate_json",
        "description": "检查字符串是否为合法 JSON。",
        "arg_names": ["payload"],
        "sync": True,
        "persist": False,
    },
    {
        "name": "fix_json",
        "description": "利用 LLM 对接近 JSON 但不合法的文本进行纠错。",
        "arg_names": ["raw_text", "instruction"],
        "sync": False,
        "persist": False,
    },
    {
        "name": "request_travel_info",
        "description": "当用户的旅行需求信息不完整时调用此工具，弹出表单让用户补充目的地、天数、预算、偏好等信息。只在确实缺少关键信息时调用。",
        "arg_names": ["missing_fields", "custom_message"],
        "sync": True,
    },
]
