"""
Tool 元数据管理与运行时筛选。

NodeManager 不执行工具，它只负责：
- 按工具名称映射到 requirement/research/planning/risk/render 层；
- 为工具附加场景标签；
- 根据当前消息中的关键词裁剪每层可见工具集合。

场景识别目前是轻量规则，不是语义分类模型，因此应视为优化策略，
不能替代关键业务校验。
"""

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
    """根据命名约定推断工具所属阶段，允许调用方通过 map 覆盖默认值。"""
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


def _extract_text_from_messages(messages: List[BaseMessage]) -> str:
    """统一拍平 LangChain 的字符串/内容块消息，供关键词规则扫描。"""
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
    """从用户文本推断人群、预算、偏好和行程长度等场景标签。"""
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
    """判断工具是否应在当前场景中保留；无场景标签时不做裁剪。"""
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
    管理 Agent 可用工具及其层级/场景元数据。

    ``tool_layer_map`` 和 ``tool_tags_map`` 可由外部显式传入；未传入时
    才使用本模块的命名规则自动生成。
    """

    tools: List[BaseTool] = field(default_factory=list)
    tool_layer_map: Dict[str, str] = field(default_factory=dict)
    tool_tags_map: Dict[str, Set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # dataclass 初始化完成后补齐派生元数据，避免调用方重复维护。
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
        """先识别场景，再返回分层后的动态工具白名单。"""
        tags = self.infer_scenario_tags_from_messages(messages)
        return self.grouped_tools(scenario_tags=tags)

