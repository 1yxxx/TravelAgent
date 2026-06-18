"""
五层旅行 Agent 编排器。

把一次旅行规划拆成 requirement→research→planning→risk→render，
每层只暴露该阶段工具，层失败时回滚重试。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from travel_agent.utils.logging import logger
from travel_agent.orchestration.layer_policy import LAYER_ORDER, LayerPolicy
from travel_agent.orchestration.layer_validator import LayerValidator
from travel_agent.orchestration.layer_metrics import LayerTrace, compute_layer_metrics
from travel_agent.orchestration.scenario_tags import infer_scenario_tags, _extract_text_from_messages


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


class LayeredTravelAgent:
    """五层编排包装器。"""

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

    async def ainvoke(
        self,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        messages: List[Any] = list(inputs.get("messages") or [])
        traces: List[LayerTrace] = []
        scenario_tags = infer_scenario_tags(_extract_text_from_messages(messages))

        current_tools_by_layer = self.tools_by_layer
        if self.runtime_tool_selector is not None:
            try:
                selected = self.runtime_tool_selector(messages)
                if isinstance(selected, dict):
                    current_tools_by_layer = selected
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[LayeredAgent] runtime tool selection failed: %s", exc
                )

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
                    model=self.llm, tools=allowed_tools, prompt=layer_prompt,
                )

                try:
                    result = await layer_agent.ainvoke({"messages": messages}, config=config)
                except Exception as exc:  # noqa: BLE001
                    fail_reason = f"层执行异常: {exc}"
                    logger.warning("[LayeredAgent] layer=%s attempt=%s failed: %s", layer, attempt, exc)
                    if attempt > self.policy.max_retries_per_layer:
                        break
                    messages = list(checkpoints[last_ok_key])
                    messages.append(HumanMessage(content=f"阶段 {layer} 执行失败，请重试。"))
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
                    traces.append(LayerTrace(layer=layer, attempt=attempt, success=True, reason=reason, tools_called=tool_calls))
                    break

                fail_reason = reason
                traces.append(LayerTrace(layer=layer, attempt=attempt, success=False, reason=reason, tools_called=tool_calls))
                if attempt > self.policy.max_retries_per_layer:
                    break

                logger.info("[LayeredAgent] rollback layer=%s attempt=%s reason=%s", layer, attempt, reason)
                messages = list(checkpoints[last_ok_key])
                messages.append(HumanMessage(content=f"阶段 {layer} 校验失败，原因：{reason}。请重试。"))

            if not success:
                logger.warning("[LayeredAgent] layer=%s failed after retries, reason=%s", layer, fail_reason)
                break

        metrics = compute_layer_metrics(traces)
        logger.info(
            "[LayeredAgent] metrics hit_rate=%.3f rollback_rate=%.3f attempts=%s rollbacks=%s",
            metrics.layer_hit_rate, metrics.rollback_rate, metrics.total_attempts, metrics.rollback_count,
        )

        return {
            "messages": messages,
            "layer_trace": [
                {"layer": t.layer, "attempt": t.attempt, "success": t.success, "reason": t.reason, "tools_called": t.tools_called}
                for t in traces
            ],
            "layer_metrics": asdict(metrics),
        }
