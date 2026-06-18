"""
Tool 元数据管理与运行时筛选。

NodeManager 不执行工具，它只负责：
- 按工具名称映射到 requirement/research/planning/risk/render 层；
- 为工具附加场景标签；
- 根据当前消息中的关键词裁剪每层可见工具集合。

与原先版本的关键区别：场景标签推断逻辑统一从
``travel_agent.orchestration.scenario_tags`` 导入，消除了与
``layered_agent.py`` 的重复代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from travel_agent.orchestration.layer_policy import LAYER_ORDER
from travel_agent.orchestration.scenario_tags import infer_scenario_tags, _extract_text_from_messages


def _default_layer_for_tool(tool_name: str) -> str:
    """根据命名约定推断工具所属阶段。"""
    name = (tool_name or "").strip().lower()

    if name in {"validate_json", "fix_json"}:
        return "risk"

    if name.startswith("render_"):
        return "render"

    if name in {"smart_plan_itinerary", "plan_itinerary", "format_itinerary", "estimate_budget"}:
        return "planning"

    if (
        name.startswith("search_")
        or name in {"check_weather", "recommend_transport", "plan_route"}
    ):
        return "research"

    return "planning"


def _default_tags_for_tool(tool_name: str) -> Set[str]:
    """为工具生成用于场景过滤的默认能力标签。"""
    name = (tool_name or "").strip().lower()

    if name in {"validate_json", "fix_json"}:
        return {"risk", "general"}

    if name.startswith("render_"):
        return {"render", "general"}

    if name in {"check_weather"}:
        return {"risk", "general"}

    if name in {"recommend_transport", "plan_route"}:
        return {"transport", "long_trip", "general"}

    if name in {"search_hotel"}:
        return {"accommodation", "general"}

    if name in {"search_restaurant"}:
        return {"food", "general"}

    if name in {"estimate_budget"}:
        return {"budget", "general"}

    if name in {"search_poi"}:
        return {"sightseeing", "general"}

    if name in {"smart_plan_itinerary", "plan_itinerary", "format_itinerary"}:
        return {"planning", "custom", "general"}

    return {"general"}


def _tool_matches_scenario(tool_tags: Set[str], scenario_tags: Set[str]) -> bool:
    """判断工具是否应在当前场景中保留。"""
    if not scenario_tags:
        return True

    if tool_tags.intersection({"general", "risk", "render"}):
        return True

    expanded_tags = set(scenario_tags)
    if "luxury" in scenario_tags or "budget" in scenario_tags:
        expanded_tags.update({"accommodation", "food", "planning", "budget"})
    if "family" in scenario_tags or "senior" in scenario_tags:
        expanded_tags.update({"transport", "accommodation", "planning"})
    if "long_trip" in scenario_tags or "short_trip" in scenario_tags:
        expanded_tags.update({"transport", "planning"})

    return bool(tool_tags.intersection(expanded_tags))


@dataclass
class NodeManager:
    """管理 Agent 可用工具及其层级/场景元数据。"""

    tools: List[BaseTool] = field(default_factory=list)
    tool_layer_map: Dict[str, str] = field(default_factory=dict)
    tool_tags_map: Dict[str, Set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tool_layer_map:
            self.tool_layer_map = {
                t.name: _default_layer_for_tool(t.name)
                for t in self.tools
            }
        if not self.tool_tags_map:
            self.tool_tags_map = {
                t.name: _default_tags_for_tool(t.name)
                for t in self.tools
            }

    def as_dict(self) -> Dict[str, BaseTool]:
        return {t.name: t for t in self.tools}

    def infer_scenario_tags_from_messages(self, messages: List[BaseMessage]) -> Set[str]:
        return infer_scenario_tags(_extract_text_from_messages(messages))

    def get_tools_by_layer(
        self, layer: str, *, scenario_tags: Set[str] | None = None
    ) -> List[BaseTool]:
        tags = scenario_tags or set()
        return [
            t for t in self.tools
            if self.tool_layer_map.get(t.name) == layer
            and _tool_matches_scenario(self.tool_tags_map.get(t.name, {"general"}), tags)
        ]

    def grouped_tools(
        self, *, scenario_tags: Set[str] | None = None
    ) -> Dict[str, List[BaseTool]]:
        return {
            layer: self.get_tools_by_layer(layer, scenario_tags=scenario_tags)
            for layer in LAYER_ORDER
        }

    def grouped_tools_for_messages(
        self, messages: List[BaseMessage]
    ) -> Dict[str, List[BaseTool]]:
        tags = self.infer_scenario_tags_from_messages(messages)
        return self.grouped_tools(scenario_tags=tags)
