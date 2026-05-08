from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from travel_agent.utils.logging import logger


LAYER_ORDER: List[str] = [
    "requirement",
    "research",
    "planning",
    "risk",
    "render",
]


def _make_layer_prompt(base_prompt: str, layer_name: str, allowed_tool_names: List[str]) -> str:
    allowed_text = ", ".join(allowed_tool_names) if allowed_tool_names else "(none)"
    return (
        f"{base_prompt}\n\n"
        "## 执行阶段约束\n"
        f"当前阶段: {layer_name}\n"
        "你必须仅使用当前阶段允许的工具；如果当前阶段无需工具，请直接给出结构化思考结论后进入下一阶段。\n"
        f"允许工具: {allowed_text}\n"
    )


def _collect_tool_calls(messages: List[Any]) -> List[str]:
    names: List[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", []) or []:
                name = tc.get("name")
                if name:
                    names.append(name)
    return names


def _extract_text(messages: List[Any]) -> str:
    parts: List[str] = []
    for m in messages:
        content = getattr(m, "content", "") or ""
        if isinstance(content, list):
            parts.extend((c.get("text") or "") if isinstance(c, dict) else str(c) for c in content)
        else:
            parts.append(str(content))
    return "\n".join(parts)


def _infer_scenario_tags(messages: List[Any]) -> Set[str]:
    t = _extract_text(messages).lower()
    tags: Set[str] = set()
    if any(k in t for k in ("老人", "老年", "无障碍", "轮椅")):
        tags.add("senior")
    if any(k in t for k in ("亲子", "小孩", "儿童", "带娃", "家庭")):
        tags.add("family")
    if any(k in t for k in ("长线", "环线", "跨省", "多城", "7天", "10天")):
        tags.add("long_trip")
    return tags


@dataclass
class LayerPolicy:
    enabled: bool = False
    max_retries_per_layer: int = 1
    strict_validation: bool = False


@dataclass
class LayerTrace:
    layer: str
    attempt: int
    success: bool
    reason: str = ""
    tools_called: List[str] = field(default_factory=list)


@dataclass
class LayerMetrics:
    attempted_layers: int
    successful_layers: int
    layer_hit_rate: float
    total_attempts: int
    rollback_count: int
    rollback_rate: float
    per_layer_attempts: Dict[str, int] = field(default_factory=dict)
    per_layer_success: Dict[str, bool] = field(default_factory=dict)


def compute_layer_metrics(traces: List[LayerTrace]) -> LayerMetrics:
    attempts = [t for t in traces if t.attempt > 0]
    per_layer_attempts: Dict[str, int] = {}
    per_layer_success: Dict[str, bool] = {}

    for t in attempts:
        per_layer_attempts[t.layer] = per_layer_attempts.get(t.layer, 0) + 1
        per_layer_success[t.layer] = per_layer_success.get(t.layer, False) or t.success

    attempted_layers = len(per_layer_attempts)
    successful_layers = sum(1 for ok in per_layer_success.values() if ok)
    total_attempts = len(attempts)
    rollback_count = sum(1 for t in attempts if not t.success)

    layer_hit_rate = (
        successful_layers / attempted_layers if attempted_layers > 0 else 0.0
    )
    rollback_rate = (
        rollback_count / total_attempts if total_attempts > 0 else 0.0
    )

    return LayerMetrics(
        attempted_layers=attempted_layers,
        successful_layers=successful_layers,
        layer_hit_rate=layer_hit_rate,
        total_attempts=total_attempts,
        rollback_count=rollback_count,
        rollback_rate=rollback_rate,
        per_layer_attempts=per_layer_attempts,
        per_layer_success=per_layer_success,
    )


class LayerValidator:
    """
    最小可行层校验：
    - research 层至少应产生一次工具调用（通常为搜索/天气类）
    - planning 层至少应产生一次规划/编排调用
    - risk / render 在 strict_validation 时必须产生工具调用
    """

    _RESEARCH_HINTS = ("search_", "weather", "recommend_transport", "plan_route")
    _PLANNING_HINTS = ("plan_", "format_itinerary", "smart_plan_itinerary", "estimate_budget")

    def __init__(self, strict: bool = False):
        self.strict = strict

    def validate(self, layer: str, tool_calls: List[str], scenario_tags: Optional[Set[str]] = None) -> tuple[bool, str]:
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


class LayeredTravelAgent:
    """
    五层编排包装器：
    1) requirement
    2) research
    3) planning
    4) risk
    5) render

    每层只暴露该层工具，层失败时从最近通过校验的 checkpoint 回滚并重试。
    """

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        tools_by_layer: Dict[str, List[BaseTool]],
        base_system_prompt: str,
        policy: LayerPolicy,
        runtime_tool_selector: Optional[Callable[[List[Any]], Dict[str, List[BaseTool]]]] = None,
    ) -> None:
        self.llm = llm
        self.tools_by_layer = tools_by_layer
        self.base_system_prompt = base_system_prompt
        self.policy = policy
        self.validator = LayerValidator(strict=policy.strict_validation)
        self.runtime_tool_selector = runtime_tool_selector

    async def ainvoke(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        messages: List[Any] = list(inputs.get("messages") or [])
        traces: List[LayerTrace] = []
        scenario_tags = _infer_scenario_tags(messages)

        current_tools_by_layer = self.tools_by_layer
        if self.runtime_tool_selector is not None:
            try:
                selected = self.runtime_tool_selector(messages)
                if isinstance(selected, dict):
                    current_tools_by_layer = selected
            except Exception as exc:  # noqa: BLE001
                logger.warning("[LayeredAgent] runtime tool selection failed, fallback to default: %s", exc)

        checkpoints: Dict[str, List[Any]] = {"start": list(messages)}
        last_ok_key = "start"

        for layer in LAYER_ORDER:
            allowed_tools = current_tools_by_layer.get(layer, [])
            allowed_names = [t.name for t in allowed_tools]

            if not allowed_tools:
                traces.append(LayerTrace(layer=layer, attempt=0, success=True, reason="no-tools"))
                checkpoints[layer] = list(messages)
                last_ok_key = layer
                continue

            layer_prompt = _make_layer_prompt(self.base_system_prompt, layer, allowed_names)

            attempt = 0
            success = False
            fail_reason = ""

            while attempt <= self.policy.max_retries_per_layer:
                attempt += 1
                before_len = len(messages)

                layer_agent = create_react_agent(
                    model=self.llm,
                    tools=allowed_tools,
                    prompt=layer_prompt,
                )

                try:
                    result = await layer_agent.ainvoke({"messages": messages}, config=config)
                except Exception as exc:  # noqa: BLE001
                    fail_reason = f"层执行异常: {exc}"
                    logger.warning("[LayeredAgent] layer=%s attempt=%s failed: %s", layer, attempt, exc)
                    if attempt > self.policy.max_retries_per_layer:
                        break
                    messages = list(checkpoints[last_ok_key])
                    messages.append(
                        HumanMessage(
                            content=(
                                f"上一轮在阶段 {layer} 执行失败，请在该阶段重试，"
                                "优先保证输出可被后续阶段消费。"
                            )
                        )
                    )
                    continue

                messages = list(result.get("messages") or messages)
                layer_delta = messages[before_len:]
                tool_calls = _collect_tool_calls(layer_delta)

                ok, reason = self.validator.validate(layer=layer, tool_calls=tool_calls, scenario_tags=scenario_tags)
                if ok:
                    success = True
                    fail_reason = reason
                    checkpoints[layer] = list(messages)
                    last_ok_key = layer
                    traces.append(
                        LayerTrace(
                            layer=layer,
                            attempt=attempt,
                            success=True,
                            reason=reason,
                            tools_called=tool_calls,
                        )
                    )
                    break

                fail_reason = reason
                traces.append(
                    LayerTrace(
                        layer=layer,
                        attempt=attempt,
                        success=False,
                        reason=reason,
                        tools_called=tool_calls,
                    )
                )

                if attempt > self.policy.max_retries_per_layer:
                    break

                logger.info(
                    "[LayeredAgent] rollback layer=%s attempt=%s reason=%s checkpoint=%s",
                    layer,
                    attempt,
                    reason,
                    last_ok_key,
                )
                messages = list(checkpoints[last_ok_key])
                messages.append(
                    HumanMessage(
                        content=(
                            f"阶段 {layer} 校验失败，原因：{reason}。"
                            "请在当前阶段重新执行，并确保满足最小完备约束。"
                        )
                    )
                )

            if not success:
                logger.warning(
                    "[LayeredAgent] layer=%s failed after retries, reason=%s",
                    layer,
                    fail_reason,
                )
                break

        metrics = compute_layer_metrics(traces)
        logger.info(
            "[LayeredAgent] metrics hit_rate=%.3f rollback_rate=%.3f attempts=%s rollbacks=%s",
            metrics.layer_hit_rate,
            metrics.rollback_rate,
            metrics.total_attempts,
            metrics.rollback_count,
        )

        return {
            "messages": messages,
            "layer_trace": [
                {
                    "layer": t.layer,
                    "attempt": t.attempt,
                    "success": t.success,
                    "reason": t.reason,
                    "tools_called": t.tools_called,
                }
                for t in traces
            ],
            "layer_metrics": asdict(metrics),
        }
