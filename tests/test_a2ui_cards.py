"""Tests for A2UI card generation — form_card payloads and place_card extraction."""

from __future__ import annotations

import json

from langchain_core.messages import ToolMessage

from travel_agent.a2ui_cards import build_form_card_payload, extract_place_cards


class TestBuildFormCardPayload:
    def test_missing_destination_puts_destination_first(self):
        precheck = {"ok": False, "reason": "missing_destination", "suggestion": "请先告诉我目的地。"}
        payload = build_form_card_payload(precheck)
        assert payload["type"] == "form_card"
        assert payload["id"] == "precheck_missing_destination"
        assert payload["title"] == "目的地信息"
        assert payload["fields"][0]["key"] == "destination"
        assert payload["fields"][0]["required"] is True

    def test_missing_duration_puts_days_first(self):
        precheck = {"ok": False, "reason": "missing_duration", "suggestion": "请补充出行天数。"}
        payload = build_form_card_payload(precheck)
        assert payload["id"] == "precheck_missing_duration"
        assert payload["title"] == "出行天数"
        assert payload["fields"][0]["key"] == "days"

    def test_query_too_short_shows_all_fields(self):
        precheck = {"ok": False, "reason": "query_too_short", "suggestion": "请补充。"}
        payload = build_form_card_payload(precheck)
        keys = [f["key"] for f in payload["fields"]]
        assert "destination" in keys
        assert "days" in keys
        assert "budget" in keys
        assert "preference" in keys
        assert payload["fields"][0]["required"] is True

    def test_intent_unclear_excludes_budget_field(self):
        precheck = {"ok": False, "reason": "intent_unclear", "suggestion": "告诉我你的旅行需求。"}
        payload = build_form_card_payload(precheck)
        keys = [f["key"] for f in payload["fields"]]
        assert "budget" not in keys

    def test_unknown_reason_falls_back_to_default(self):
        precheck = {"ok": False, "reason": "something_else", "suggestion": ""}
        payload = build_form_card_payload(precheck)
        assert payload["type"] == "form_card"
        assert len(payload["fields"]) == 4

    def test_form_card_includes_message_from_suggestion(self):
        precheck = {"ok": False, "reason": "missing_destination", "suggestion": "请先告诉我目的地城市。"}
        payload = build_form_card_payload(precheck)
        assert "请先告诉我目的地城市" in payload["message"]


class TestExtractPlaceCards:
    def _make_tool_message(self, content, tool_call_id="call_abc", tool_name="search_poi"):
        return ToolMessage(content=content, tool_call_id=tool_call_id), tool_name

    def _make_mcp_envelope(self, result, is_error=False):
        return json.dumps({"artifact_id": "art_001", "result": result, "isError": is_error})

    def test_search_poi_generates_place_card(self):
        poi_list = [
            {"name": "故宫", "longitude": 116.397, "latitude": 39.918,
             "address": "北京东城", "rating": "4.8", "cost": "60",
             "photos": [], "id": "B001", "cityname": "北京"},
        ]
        msg = ToolMessage(
            content=self._make_mcp_envelope(poi_list),
            tool_call_id="call_poi",
        )
        call_id_to_name = {"call_poi": "search_poi"}
        cards = extract_place_cards([msg], call_id_to_name)

        assert len(cards) == 1
        card = cards[0]
        assert card["type"] == "place_card"
        assert card["source_tool"] == "search_poi"
        assert card["category"] == "poi"
        assert card["city"] == "北京"
        assert len(card["places"]) == 1
        assert card["places"][0]["name"] == "故宫"
        assert card["places"][0]["rating"] == "4.8"
        assert card["places"][0]["longitude"] == 116.397
        assert card["places"][0]["latitude"] == 39.918

    def test_search_hotel_generates_hotel_card(self):
        hotel_list = [
            {"name": "锦江之星", "longitude": 104.06, "latitude": 30.67,
             "address": "成都锦江", "rating": "4.2", "cost": "280", "photos": [], "id": "H001"},
        ]
        msg = ToolMessage(
            content=self._make_mcp_envelope(hotel_list),
            tool_call_id="call_hotel",
        )
        cards = extract_place_cards([msg], {"call_hotel": "search_hotel"})
        assert len(cards) == 1
        assert cards[0]["category"] == "hotel"
        assert cards[0]["category_label"] == "🏨 酒店"
        assert cards[0]["places"][0]["name"] == "锦江之星"

    def test_search_restaurant_generates_restaurant_card(self):
        res_list = [
            {"name": "海底捞", "longitude": 116.39, "latitude": 39.92,
             "cuisine": "火锅", "rating": "4.5", "cost": "120", "photos": [], "id": "R001"},
        ]
        msg = ToolMessage(
            content=self._make_mcp_envelope(res_list),
            tool_call_id="call_res",
        )
        cards = extract_place_cards([msg], {"call_res": "search_restaurant"})
        assert len(cards) == 1
        assert cards[0]["category"] == "restaurant"

    def test_non_search_tool_ignored(self):
        msg = ToolMessage(content="{}", tool_call_id="call_xyz")
        cards = extract_place_cards([msg], {"call_xyz": "plan_route"})
        assert cards == []

    def test_error_result_skipped(self):
        msg = ToolMessage(
            content=self._make_mcp_envelope("error", is_error=True),
            tool_call_id="call_err",
        )
        cards = extract_place_cards([msg], {"call_err": "search_poi"})
        assert cards == []

    def test_same_tool_deduplicated(self):
        poi_list = [{"name": "A", "longitude": 116, "latitude": 39, "photos": [], "id": "1"}]
        msg1 = ToolMessage(content=self._make_mcp_envelope(poi_list), tool_call_id="call_1")
        msg2 = ToolMessage(content=self._make_mcp_envelope(poi_list), tool_call_id="call_2")
        cards = extract_place_cards(
            [msg1, msg2],
            {"call_1": "search_poi", "call_2": "search_poi"},
        )
        assert len(cards) == 1

    def test_max_places_limit(self):
        many_pois = [
            {"name": f"POI_{i}", "longitude": 116 + i * 0.01, "latitude": 39 + i * 0.01,
             "photos": [], "id": str(i)}
            for i in range(10)
        ]
        msg = ToolMessage(
            content=self._make_mcp_envelope(many_pois),
            tool_call_id="call_many",
        )
        cards = extract_place_cards([msg], {"call_many": "search_poi"}, max_places=3)
        assert len(cards[0]["places"]) == 3

    def test_photo_extraction(self):
        poi = [{
            "name": "景点", "longitude": 116, "latitude": 39, "id": "P001",
            "photos": [
                {"url": "https://example.com/img1.jpg"},
                {"url": "https://example.com/img2.jpg"},
            ],
        }]
        msg = ToolMessage(
            content=self._make_mcp_envelope(poi),
            tool_call_id="call_photo",
        )
        cards = extract_place_cards([msg], {"call_photo": "search_poi"})
        assert len(cards[0]["places"][0]["photos"]) == 2
        assert cards[0]["places"][0]["photos"][0] == "https://example.com/img1.jpg"

    def test_photos_wrapped_in_photo_key(self):
        poi = [{
            "name": "景点", "longitude": 116, "latitude": 39, "id": "P002",
            "photos": {"photo": [{"url": "https://example.com/p.jpg"}]},
        }]
        msg = ToolMessage(
            content=self._make_mcp_envelope(poi),
            tool_call_id="call_wrapped",
        )
        cards = extract_place_cards([msg], {"call_wrapped": "search_poi"})
        assert len(cards[0]["places"][0]["photos"]) == 1

    def test_non_http_photos_filtered(self):
        poi = [{
            "name": "景点", "longitude": 116, "latitude": 39, "id": "P003",
            "photos": [{"url": "ftp://bad.example/img.jpg"}, {"url": "/relative/path.jpg"}],
        }]
        msg = ToolMessage(
            content=self._make_mcp_envelope(poi),
            tool_call_id="call_filter",
        )
        cards = extract_place_cards([msg], {"call_filter": "search_poi"})
        assert cards[0]["places"][0]["photos"] == []

    def test_missing_coordinates_set_to_none(self):
        poi = [{"name": "无坐标景点", "photos": [], "id": "NC"}]
        msg = ToolMessage(
            content=self._make_mcp_envelope(poi),
            tool_call_id="call_nocoord",
        )
        cards = extract_place_cards([msg], {"call_nocoord": "search_poi"})
        assert cards[0]["places"][0]["longitude"] is None
        assert cards[0]["places"][0]["latitude"] is None

    def test_summary_line(self):
        poi = [{
            "name": "餐厅", "longitude": 116, "latitude": 39, "id": "R",
            "rating": "4.7", "cost": "88", "photos": [],
        }]
        msg = ToolMessage(
            content=self._make_mcp_envelope(poi),
            tool_call_id="call_summary",
        )
        cards = extract_place_cards([msg], {"call_summary": "search_restaurant"})
        summary = cards[0]["places"][0]["summary"]
        assert "⭐4.7" in summary
        assert "¥88" in summary

    def test_content_list_format_from_mcp_adapter(self):
        poi_list = [{"name": "天安门", "longitude": 116.397, "latitude": 39.909, "photos": [], "id": "TAM"}]
        raw = json.dumps(poi_list)
        mcp_wrapped = [{"type": "text", "text": self._make_mcp_envelope(raw)}]
        msg = ToolMessage(content=mcp_wrapped, tool_call_id="call_list")
        cards = extract_place_cards([msg], {"call_list": "search_poi"})
        assert len(cards) == 1
        assert cards[0]["places"][0]["name"] == "天安门"
