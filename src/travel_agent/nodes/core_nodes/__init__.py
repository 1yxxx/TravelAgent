"""
旅行 Agent 的核心原子工具包。

这些函数使用 LangChain ``@tool`` 装饰，可直接注入 Agent，也可由 MCP
Wrapper 继续封装。这里仅导出常用工具，完整注册列表以
``mcp/register_tools.py`` 为准。
"""
from travel_agent.nodes.core_nodes.search_poi import search_poi_tool
from travel_agent.nodes.core_nodes.check_weather import check_weather_tool
from travel_agent.nodes.core_nodes.search_hotel import search_hotel_tool
from travel_agent.nodes.core_nodes.search_restaurant import search_restaurant_tool
from travel_agent.nodes.core_nodes.plan_route import plan_route_tool
from travel_agent.nodes.core_nodes.format_itinerary import format_itinerary_tool

__all__ = [
    "search_poi_tool",
    "check_weather_tool",
    "search_hotel_tool",
    "search_restaurant_tool",
    "plan_route_tool",
    "format_itinerary_tool",
]
