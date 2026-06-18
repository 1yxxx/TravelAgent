from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest import TestCase

from langchain_core.messages import HumanMessage

from travel_agent.agent.node_manager import NodeManager


class TestToolScenarioFilter(TestCase):
    def test_infer_scenario_tags_from_messages(self):
        manager = NodeManager(
            tools=cast(
                Any,
                [
                    SimpleNamespace(name="search_poi"),
                    SimpleNamespace(name="search_hotel"),
                    SimpleNamespace(name="search_restaurant"),
                ],
            )
        )

        tags = manager.infer_scenario_tags_from_messages(
            [HumanMessage(content="带老人去成都，预算低，周末短途")]
        )

        self.assertIn("senior", tags)
        self.assertIn("budget", tags)
        self.assertIn("short_trip", tags)

    def test_grouped_tools_for_messages_filters_by_scenario(self):
        tools = [
            SimpleNamespace(name="search_poi"),
            SimpleNamespace(name="search_hotel"),
            SimpleNamespace(name="search_restaurant"),
            SimpleNamespace(name="recommend_transport"),
            SimpleNamespace(name="check_weather"),
            SimpleNamespace(name="render_itinerary"),
        ]
        manager = NodeManager(tools=cast(Any, tools))

        grouped = manager.grouped_tools_for_messages(
            [HumanMessage(content="情侣轻奢定制旅行，关注住宿和餐厅")]
        )
        research_names = {t.name for t in grouped["research"]}
        render_names = {t.name for t in grouped["render"]}

        # 通用工具仍存在
        self.assertIn("check_weather", research_names)
        self.assertIn("render_itinerary", render_names)
        # 场景相关工具被保留
        self.assertIn("search_hotel", research_names)
        self.assertIn("search_restaurant", research_names)
