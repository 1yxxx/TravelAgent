from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest import TestCase

from langchain_core.messages import HumanMessage

from travel_agent.agent.context import ClientContext


class TestPrecheck(TestCase):
    def _make_context(self) -> ClientContext:
        cfg = SimpleNamespace(
            memory_switch=SimpleNamespace(
                enabled=True,
                soft_message_threshold=24,
                hard_message_threshold=60,
                soft_token_threshold=3500,
                hard_token_threshold=7000,
            )
        )
        return ClientContext(
            cfg=cast(Any, cfg),
            session_id="s_precheck",
            node_manager=cast(Any, SimpleNamespace()),
            _base_system_prompt="base",
        )

    def test_precheck_missing_destination(self):
        ctx = self._make_context()
        result = ctx.precheck_user_request([HumanMessage(content="我想旅行3天")])
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "missing_destination")

    def test_precheck_missing_duration(self):
        ctx = self._make_context()
        result = ctx.precheck_user_request([HumanMessage(content="帮我规划成都行程")])
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "missing_duration")

    def test_precheck_ok(self):
        ctx = self._make_context()
        result = ctx.precheck_user_request([HumanMessage(content="帮我规划成都3天亲子游，预算5000")])
        self.assertTrue(result["ok"])
