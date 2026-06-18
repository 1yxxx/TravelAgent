"""
层执行结果的最小可行校验器。

负责检查每一层的工具调用是否满足最小完备约束：
- research 层至少应产生一次搜索/天气类工具调用；
- planning 层至少应产生一次规划/编排类调用；
- 特殊场景（老人/家庭/长线）会有额外校验。
"""

from __future__ import annotations

from typing import List, Optional, Set


class LayerValidator:
    """最小可行层校验器。"""

    _RESEARCH_HINTS = ("search_", "weather", "recommend_transport", "plan_route")
    _PLANNING_HINTS = ("plan_", "format_itinerary", "smart_plan_itinerary", "estimate_budget")

    def __init__(self, strict: bool = False):
        self.strict = strict

    def validate(
        self,
        layer: str,
        tool_calls: List[str],
        scenario_tags: Optional[Set[str]] = None,
    ) -> tuple[bool, str]:
        """校验当前层的工具调用是否满足最小完备约束。

        Returns:
            (ok, reason)
        """
        scenario_tags = scenario_tags or set()

        if layer == "requirement":
            return True, "ok"

        if layer == "research":
            if any(any(h in name for h in self._RESEARCH_HINTS) for name in tool_calls):
                if scenario_tags.intersection({"senior", "family", "long_trip"}) and "check_weather" not in tool_calls:
                    return False, "research 层缺少天气校验（特殊人群/长线场景）"
                return True, "ok"
            return False, "research 层未触发检索/调研类工具"

        if layer == "planning":
            if any(any(h in name for h in self._PLANNING_HINTS) for name in tool_calls):
                if "long_trip" in scenario_tags and not any(n in tool_calls for n in ("recommend_transport", "plan_route")):
                    return False, "planning 层缺少交通路径校验（长线场景）"
                return True, "ok"
            return False, "planning 层未触发行程编排类工具"

        if layer in {"risk", "render"}:
            if not self.strict:
                return True, "ok"
            if tool_calls:
                return True, "ok"
            return False, f"{layer} 层在 strict_validation 下未触发工具"

        return True, "ok"
