from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase

from langchain_core.messages import HumanMessage, SystemMessage

from travel_agent.agent import ClientContext, choose_memory_framework


class TestMemoryFramework(IsolatedAsyncioTestCase):
    async def test_choose_memory_framework_thresholds(self):
        mode, reason = choose_memory_framework(
            enabled=True,
            message_count=10,
            token_estimate=1000,
            soft_message_threshold=24,
            hard_message_threshold=60,
            soft_token_threshold=3500,
            hard_token_threshold=7000,
        )
        self.assertEqual(mode, "full_context")
        self.assertEqual(reason, "normal")

        mode, reason = choose_memory_framework(
            enabled=True,
            message_count=30,
            token_estimate=1000,
            soft_message_threshold=24,
            hard_message_threshold=60,
            soft_token_threshold=3500,
            hard_token_threshold=7000,
        )
        self.assertEqual(mode, "compressed_context")
        self.assertEqual(reason, "soft-threshold")

        mode, reason = choose_memory_framework(
            enabled=True,
            message_count=70,
            token_estimate=1000,
            soft_message_threshold=24,
            hard_message_threshold=60,
            soft_token_threshold=3500,
            hard_token_threshold=7000,
        )
        self.assertEqual(mode, "profile_only")
        self.assertEqual(reason, "hard-threshold")

    async def test_prepare_messages_injects_dynamic_system(self):
        cfg = SimpleNamespace(
            memory_switch=SimpleNamespace(
                enabled=True,
                soft_message_threshold=2,
                hard_message_threshold=4,
                soft_token_threshold=100,
                hard_token_threshold=200,
            )
        )

        ctx = ClientContext(
            cfg=cast(Any, cfg),
            session_id="s1",
            node_manager=cast(Any, SimpleNamespace()),
            lang="zh",
            mcp_client=None,
            memory_compressor=None,
            user_profile=None,
            artifact_store=None,
            _base_system_prompt="base prompt",
        )

        invoke_messages, meta = await ctx.prepare_messages_for_invoke(
            [HumanMessage(content="帮我规划周末行程")]
        )

        self.assertIsInstance(invoke_messages[0], SystemMessage)
        self.assertIn("base prompt", invoke_messages[0].content)
        self.assertEqual(meta["mode"], "full_context")

    async def test_quality_feedback_from_layer_trace(self):
        cfg = SimpleNamespace(
            memory_switch=SimpleNamespace(
                enabled=True,
                soft_message_threshold=24,
                hard_message_threshold=60,
                soft_token_threshold=3500,
                hard_token_threshold=7000,
            )
        )
        ctx = ClientContext(
            cfg=cast(Any, cfg),
            session_id="s2",
            node_manager=cast(Any, SimpleNamespace()),
            _base_system_prompt="base",
        )

        ok_feedback = ctx.build_quality_feedback({"layer_trace": [{"layer": "research", "success": True, "attempt": 1}]})
        self.assertEqual(ok_feedback["status"], "ok")

        bad_feedback = ctx.build_quality_feedback(
            {
                "layer_trace": [
                    {"layer": "research", "success": False, "attempt": 1, "reason": "missing tool"}
                ]
            }
        )
        self.assertEqual(bad_feedback["status"], "needs_retry")
        self.assertEqual(bad_feedback["failure_count"], 1)
