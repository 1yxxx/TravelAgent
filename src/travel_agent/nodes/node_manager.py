from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool


LAYER_ORDER: List[str] = [
    "requirement",
    "research",
    "planning",
    "risk",
    "render",
]


def _default_layer_for_tool(tool_name: str) -> str:
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

    # 对未知工具默认放在 planning，避免影响渲染和风险层的约束强度。
    return "planning"


def _default_tags_for_tool(tool_name: str) -> Set[str]:
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


def _extract_text_from_messages(messages: List[BaseMessage]) -> str:
    chunks: List[str] = []
    for m in messages:
        content = getattr(m, "content", "") or ""
        if isinstance(content, list):
            text = "\n".join(
                (x.get("text") or "") if isinstance(x, dict) else str(x)
                for x in content
            )
            chunks.append(text)
        else:
            chunks.append(str(content))
    return "\n".join(chunks)


def _infer_scenario_tags(text: str) -> Set[str]:
    t = (text or "").lower()
    tags: Set[str] = set()

    if any(k in t for k in ("亲子", "小孩", "儿童", "带娃", "家庭")):
        tags.add("family")
    if any(k in t for k in ("老人", "老年", "无障碍", "轮椅")):
        tags.add("senior")
    if any(k in t for k in ("情侣", "约会", "蜜月", "浪漫")):
        tags.add("couple")
    if any(k in t for k in ("独行", "一个人", "solo", "单人")):
        tags.add("solo")

    if any(k in t for k in ("穷游", "省钱", "预算低", "性价比", "便宜")):
        tags.add("budget")
    if any(k in t for k in ("轻奢", "高端", "豪华", "luxury", "定制")):
        tags.add("luxury")

    if any(k in t for k in ("户外", "徒步", "露营", "登山", "骑行")):
        tags.add("outdoor")
    if any(k in t for k in ("人文", "博物馆", "古镇", "历史", "文化")):
        tags.add("culture")

    if any(k in t for k in ("短途", "周边", "周末", "1天", "2天")):
        tags.add("short_trip")
    if any(k in t for k in ("长线", "环线", "跨省", "多城", "5天", "7天", "10天")):
        tags.add("long_trip")

    if any(k in t for k in ("定制", "个性化", "小众", "深度游")):
        tags.add("custom")

    return tags


def _tool_matches_scenario(tool_tags: Set[str], scenario_tags: Set[str]) -> bool:
    if not scenario_tags:
        return True

    # 通用工具、风控工具、渲染工具在任何场景都保留。
    if tool_tags.intersection({"general", "risk", "render"}):
        return True

    # 同义映射：用户 "luxury" 也需要住宿、餐饮、规划、预算等核心工具。
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
    """
    负责管理所有可用的工具节点，提供按名称查询等能力。
    """

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
        return _infer_scenario_tags(_extract_text_from_messages(messages))

    def get_tools_by_layer(self, layer: str, *, scenario_tags: Set[str] | None = None) -> List[BaseTool]:
        tags = scenario_tags or set()
        return [
            t for t in self.tools
            if self.tool_layer_map.get(t.name) == layer
            and _tool_matches_scenario(self.tool_tags_map.get(t.name, {"general"}), tags)
        ]

    def grouped_tools(self, *, scenario_tags: Set[str] | None = None) -> Dict[str, List[BaseTool]]:
        return {layer: self.get_tools_by_layer(layer, scenario_tags=scenario_tags) for layer in LAYER_ORDER}

    def grouped_tools_for_messages(self, messages: List[BaseMessage]) -> Dict[str, List[BaseTool]]:
        tags = self.infer_scenario_tags_from_messages(messages)
        return self.grouped_tools(scenario_tags=tags)

