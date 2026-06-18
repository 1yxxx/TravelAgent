"""
核心工具包 —— 完整工具注册表。

按领域分为四个子包：
- ``search``:    搜索/检索 (search_poi, search_hotel, search_restaurant, check_weather)
- ``planning``:  规划/编排 (plan_itinerary, smart_plan_itinerary, plan_route, etc.)
- ``rendering``: 渲染/展示 (render_map_pois, render_map_route, render_itinerary)
- ``utility``:   横切工具 (validate_json, fix_json, request_travel_info)

``ALL_TOOLS`` 是完整的工具注册表，供 MCP Server 和 Agent 使用。
"""

from travel_agent.tools.search import (
    search_poi_tool, search_hotel_tool, search_restaurant_tool, check_weather_tool,
)
from travel_agent.tools.planning import (
    plan_itinerary_tool, smart_plan_itinerary_tool, plan_route_tool,
    estimate_budget_tool, recommend_transport_tool, format_itinerary_tool,
)
from travel_agent.tools.rendering import (
    render_map_pois_tool, render_map_route_tool, render_itinerary_tool,
)
from travel_agent.tools.utility import (
    validate_json_tool, fix_json_tool, request_travel_info_tool,
)

ALL_TOOLS = [
    search_poi_tool,
    search_hotel_tool,
    search_restaurant_tool,
    check_weather_tool,
    plan_itinerary_tool,
    smart_plan_itinerary_tool,
    plan_route_tool,
    estimate_budget_tool,
    recommend_transport_tool,
    format_itinerary_tool,
    render_map_pois_tool,
    render_map_route_tool,
    render_itinerary_tool,
    validate_json_tool,
    fix_json_tool,
    request_travel_info_tool,
]

__all__ = [
    "ALL_TOOLS",
    # search
    "search_poi_tool", "search_hotel_tool", "search_restaurant_tool", "check_weather_tool",
    # planning
    "plan_itinerary_tool", "smart_plan_itinerary_tool", "plan_route_tool",
    "estimate_budget_tool", "recommend_transport_tool", "format_itinerary_tool",
    # rendering
    "render_map_pois_tool", "render_map_route_tool", "render_itinerary_tool",
    # utility
    "validate_json_tool", "fix_json_tool", "request_travel_info_tool",
]
