from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from travel_agent.orchestration.layered_agent import (
    LayerPolicy,
    LayerTrace,
    LayerValidator,
    LayeredTravelAgent,
    compute_layer_metrics,
)


class _FakeLayerAgent:
    def __init__(self, responses):
        self._responses = responses

    async def ainvoke(self, inputs, config=None):
        messages = list(inputs.get("messages") or [])
        response = self._responses.pop(0)
        return {"messages": messages + [response]}


class TestLayerValidator(IsolatedAsyncioTestCase):
    async def test_research_layer_requires_research_tools(self):
        validator = LayerValidator(strict=False)

        ok, reason = validator.validate(layer="research", tool_calls=["search_poi"])
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

        ok, reason = validator.validate(layer="research", tool_calls=["render_itinerary"])
        self.assertFalse(ok)
        self.assertIn("未触发检索", reason)

    async def test_risk_layer_strict_mode_requires_tools(self):
        validator = LayerValidator(strict=True)

        ok, reason = validator.validate(layer="risk", tool_calls=[])
        self.assertFalse(ok)
        self.assertIn("strict_validation", reason)

    async def test_research_requires_weather_for_sensitive_scenarios(self):
        validator = LayerValidator(strict=False)
        ok, reason = validator.validate(
            layer="research",
            tool_calls=["search_poi"],
            scenario_tags={"family"},
        )
        self.assertFalse(ok)
        self.assertIn("天气校验", reason)

    async def test_planning_requires_transport_for_long_trip(self):
        validator = LayerValidator(strict=False)
        ok, reason = validator.validate(
            layer="planning",
            tool_calls=["smart_plan_itinerary"],
            scenario_tags={"long_trip"},
        )
        self.assertFalse(ok)
        self.assertIn("交通路径校验", reason)


class TestLayerMetrics(IsolatedAsyncioTestCase):
    async def test_compute_layer_metrics_hit_and_rollback_rate(self):
        traces = [
            LayerTrace(layer="requirement", attempt=0, success=True, reason="no-tools"),
            LayerTrace(layer="research", attempt=1, success=False, reason="fail"),
            LayerTrace(layer="research", attempt=2, success=True, reason="ok"),
            LayerTrace(layer="planning", attempt=1, success=True, reason="ok"),
        ]

        metrics = compute_layer_metrics(traces)

        self.assertEqual(metrics.attempted_layers, 2)
        self.assertEqual(metrics.successful_layers, 2)
        self.assertAlmostEqual(metrics.layer_hit_rate, 1.0)
        self.assertEqual(metrics.total_attempts, 3)
        self.assertEqual(metrics.rollback_count, 1)
        self.assertAlmostEqual(metrics.rollback_rate, 1 / 3)
        self.assertEqual(metrics.per_layer_attempts["research"], 2)
        self.assertTrue(metrics.per_layer_success["research"])


class TestLayeredTravelAgent(IsolatedAsyncioTestCase):
    async def test_rollback_and_metrics_output(self):
        tools_by_layer = {
            "requirement": [],
            "research": [SimpleNamespace(name="search_poi")],
            "planning": [],
            "risk": [],
            "render": [],
        }

        policy = LayerPolicy(enabled=True, max_retries_per_layer=1, strict_validation=False)
        agent = LayeredTravelAgent(
            llm=cast(Any, SimpleNamespace()),
            tools_by_layer=tools_by_layer,
            base_system_prompt="test prompt",
            policy=policy,
        )

        # 第一次研究层不触发工具 -> 触发回滚；第二次触发 search_poi -> 通过。
        responses = [
            AIMessage(content="first attempt without tool calls"),
            AIMessage(
                content="second attempt with tool call",
                tool_calls=[{"id": "c1", "name": "search_poi", "args": {}}],
            ),
        ]

        def _factory(*args, **kwargs):
            return _FakeLayerAgent(responses)

        with patch("travel_agent.orchestration.layered_agent.create_react_agent", side_effect=_factory):
            result = await agent.ainvoke({"messages": [HumanMessage(content="帮我规划行程")]})

        self.assertIn("layer_trace", result)
        self.assertIn("layer_metrics", result)

        trace = result["layer_trace"]
        research_fail = [t for t in trace if t["layer"] == "research" and t["attempt"] == 1][0]
        research_pass = [t for t in trace if t["layer"] == "research" and t["attempt"] == 2][0]
        self.assertFalse(research_fail["success"])
        self.assertTrue(research_pass["success"])

        metrics = result["layer_metrics"]
        self.assertEqual(metrics["rollback_count"], 1)
        self.assertEqual(metrics["total_attempts"], 2)
        self.assertAlmostEqual(metrics["rollback_rate"], 0.5)
        self.assertAlmostEqual(metrics["layer_hit_rate"], 1.0)

    async def test_runtime_tool_selector_takes_effect(self):
        tools_by_layer = {
            "requirement": [],
            "research": [SimpleNamespace(name="search_poi")],
            "planning": [],
            "risk": [],
            "render": [],
        }

        dynamic_tools_by_layer = {
            "requirement": [],
            "research": [SimpleNamespace(name="check_weather")],
            "planning": [],
            "risk": [],
            "render": [],
        }

        policy = LayerPolicy(enabled=True, max_retries_per_layer=0, strict_validation=False)
        agent = LayeredTravelAgent(
            llm=cast(Any, SimpleNamespace()),
            tools_by_layer=tools_by_layer,
            base_system_prompt="test prompt",
            policy=policy,
            runtime_tool_selector=lambda _msgs: dynamic_tools_by_layer,
        )

        # research 层调用 check_weather 则说明 runtime selector 生效
        responses = [
            AIMessage(
                content="runtime selected tool called",
                tool_calls=[{"id": "c1", "name": "check_weather", "args": {}}],
            ),
        ]

        def _factory(*args, **kwargs):
            return _FakeLayerAgent(responses)

        with patch("travel_agent.orchestration.layered_agent.create_react_agent", side_effect=_factory):
            result = await agent.ainvoke({"messages": [HumanMessage(content="查天气并规划行程")]})

        trace = result["layer_trace"]
        research_pass = [t for t in trace if t["layer"] == "research" and t["attempt"] == 1][0]
        self.assertTrue(research_pass["success"])
        self.assertIn("check_weather", research_pass["tools_called"])
