"""POST + SSE 聊天传输层测试。"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from travel_agent.api.chat_session import ChatSessionStore
from travel_agent.api.sse_handler import (
    ChatStreamRequest,
    create_sse_response,
    encode_sse,
    stream_chat,
)


class FakeRequest:
    def __init__(self, disconnected: bool = False) -> None:
        self.disconnected = disconnected

    async def is_disconnected(self) -> bool:
        return self.disconnected


class FakeContext:
    async def prepare_messages_for_invoke(self, messages):
        return list(messages), {
            "mode": "full_context",
            "reason": "test",
            "message_count": len(messages),
            "token_estimate": 10,
        }

    def persist_layer_metrics(self, result):
        return None

    def build_quality_feedback(self, result):
        return {"enabled": False}


class FakeAgent:
    def __init__(self) -> None:
        self.message_counts = []

    async def ainvoke(self, inputs, config=None):
        messages = list(inputs["messages"])
        self.message_counts.append(len(messages))
        user_text = next(
            message.content
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        )
        return {"messages": messages + [AIMessage(content=f"回复：{user_text}")]}


def _decode_frames(chunks: list[str]) -> list[tuple[str, dict]]:
    frames = []
    for chunk in chunks:
        if chunk.startswith(":"):
            continue
        event = ""
        data = {}
        for line in chunk.strip().splitlines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
        frames.append((event, data))
    return frames


def _collect(stream):
    async def run():
        return [chunk async for chunk in stream]

    return asyncio.run(run())


def test_encode_sse_preserves_unicode_and_frame_boundary():
    encoded = encode_sse("token", {"content": "成都\n旅行"})
    assert encoded.startswith("event: token\ndata: ")
    assert encoded.endswith("\n\n")
    assert json.loads(encoded.split("data: ", 1)[1]) == {"content": "成都\n旅行"}


def test_sse_response_headers_disable_proxy_buffering():
    async def empty_stream():
        if False:
            yield ""

    response = create_sse_response(empty_stream())
    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"


def test_session_store_returns_same_session_and_has_lock():
    async def run():
        store = ChatSessionStore(agent_builder=lambda **_: None)
        first = await store.get_or_create("same-session")
        second = await store.get_or_create("same-session")
        assert first is second
        assert isinstance(first.lock, asyncio.Lock)
        assert await store.size() == 1

    asyncio.run(run())


def test_two_post_requests_share_message_history(monkeypatch):
    agent = FakeAgent()

    async def builder(**kwargs):
        return agent, FakeContext()

    store = ChatSessionStore(agent_builder=builder)
    cfg = SimpleNamespace(a2ui=SimpleNamespace(enabled=True))
    monkeypatch.setattr(
        "travel_agent.api.sse_handler.load_settings",
        lambda _: cfg,
    )

    first = _collect(stream_chat(
        FakeRequest(),
        ChatStreamRequest(session_id="browser-session", content="第一轮"),
        "unused.toml",
        store=store,
    ))
    second = _collect(stream_chat(
        FakeRequest(),
        ChatStreamRequest(session_id="browser-session", content="第二轮"),
        "unused.toml",
        store=store,
    ))

    first_frames = _decode_frames(first)
    second_frames = _decode_frames(second)
    assert [name for name, _ in first_frames] == [
        "session", "memory_mode", "token", "done",
    ]
    assert [name for name, _ in second_frames] == [
        "session", "memory_mode", "token", "done",
    ]
    assert agent.message_counts == [1, 3]
    assert second_frames[2][1]["content"] == "回复：第二轮"


def test_form_response_uses_same_sse_path(monkeypatch):
    agent = FakeAgent()

    async def builder(**kwargs):
        return agent, FakeContext()

    store = ChatSessionStore(agent_builder=builder)
    monkeypatch.setattr(
        "travel_agent.api.sse_handler.load_settings",
        lambda _: SimpleNamespace(a2ui=SimpleNamespace(enabled=True)),
    )
    chunks = _collect(stream_chat(
        FakeRequest(),
        ChatStreamRequest(
            session_id="form-session",
            type="form_response",
            data={"destination": "成都", "days": "3", "budget": "mid"},
        ),
        "unused.toml",
        store=store,
    ))
    frames = _decode_frames(chunks)
    token = next(data["content"] for name, data in frames if name == "token")
    assert "目的地：成都" in token
    assert "天数：3天" in token
    assert "预算：舒适享受" in token
    assert frames[-1] == ("done", {"ok": True})


def test_astream_events_emits_tokens_and_tool_events_without_duplicate(monkeypatch):
    class StreamingAgent:
        async def astream_events(self, inputs, config=None, version=None):
            messages = list(inputs["messages"])
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content="成都")},
            }
            yield {
                "event": "on_tool_start",
                "name": "search_poi",
                "run_id": "tool-1",
                "data": {"input": {"city": "成都"}},
            }
            yield {
                "event": "on_tool_end",
                "name": "search_poi",
                "run_id": "tool-1",
                "data": {"output": []},
            }
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content="欢迎你")},
            }
            yield {
                "event": "on_chain_end",
                "data": {
                    "output": {
                        "messages": messages + [AIMessage(content="成都欢迎你")]
                    }
                },
            }

    async def builder(**kwargs):
        return StreamingAgent(), FakeContext()

    store = ChatSessionStore(agent_builder=builder)
    monkeypatch.setattr(
        "travel_agent.api.sse_handler.load_settings",
        lambda _: SimpleNamespace(a2ui=SimpleNamespace(enabled=False)),
    )
    frames = _decode_frames(_collect(stream_chat(
        FakeRequest(),
        ChatStreamRequest(session_id="stream-session", content="介绍成都"),
        "unused.toml",
        store=store,
    )))
    names = [name for name, _ in frames]
    assert names == [
        "session",
        "memory_mode",
        "token",
        "tool_start",
        "tool_end",
        "token",
        "done",
    ]
    assert [
        data["content"] for name, data in frames if name == "token"
    ] == ["成都", "欢迎你"]


def test_request_travel_info_is_emitted_as_a2ui_form(monkeypatch):
    class FormAgent:
        async def ainvoke(self, inputs, config=None):
            messages = list(inputs["messages"])
            call_id = "request-info-1"
            form = {
                "__a2ui_form": True,
                "id": "trip-form",
                "title": "完善信息",
                "message": "请补充目的地",
                "fields": [{"key": "destination", "field_type": "text"}],
            }
            return {
                "messages": messages + [
                    AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "request_travel_info",
                            "args": {},
                            "id": call_id,
                        }],
                    ),
                    ToolMessage(
                        content=json.dumps({"result": form}, ensure_ascii=False),
                        tool_call_id=call_id,
                    ),
                ]
            }

    async def builder(**kwargs):
        return FormAgent(), FakeContext()

    store = ChatSessionStore(agent_builder=builder)
    monkeypatch.setattr(
        "travel_agent.api.sse_handler.load_settings",
        lambda _: SimpleNamespace(a2ui=SimpleNamespace(enabled=True)),
    )
    frames = _decode_frames(_collect(stream_chat(
        FakeRequest(),
        ChatStreamRequest(session_id="a2ui-session", content="帮我旅行"),
        "unused.toml",
        store=store,
    )))
    form_event = next(data for name, data in frames if name == "a2ui")
    assert form_event["type"] == "form_card"
    assert form_event["id"] == "trip-form"
    assert any(name == "token" for name, _ in frames)
    assert frames[-1] == ("done", {"ok": True, "awaiting_form": True})


def test_map_weather_and_place_card_events_are_preserved(monkeypatch):
    agent = FakeAgent()

    async def builder(**kwargs):
        return agent, FakeContext()

    store = ChatSessionStore(agent_builder=builder)
    monkeypatch.setattr(
        "travel_agent.api.sse_handler.load_settings",
        lambda _: SimpleNamespace(a2ui=SimpleNamespace(enabled=True)),
    )
    monkeypatch.setattr(
        "travel_agent.api.sse_handler.extract_map_blocks",
        lambda *_: '```json\n{"__type":"pois","items":[]}\n```',
    )
    monkeypatch.setattr(
        "travel_agent.api.sse_handler.extract_weather_block",
        lambda *_: '```json\n{"__type":"weather","city":"成都"}\n```',
    )
    monkeypatch.setattr(
        "travel_agent.api.sse_handler.extract_place_cards",
        lambda *_: [{"type": "place_card", "places": [{"name": "宽窄巷子"}]}],
    )
    frames = _decode_frames(_collect(stream_chat(
        FakeRequest(),
        ChatStreamRequest(session_id="structured-session", content="成都景点"),
        "unused.toml",
        store=store,
    )))
    names = [name for name, _ in frames]
    assert "a2ui" in names
    assert "map_data" in names
    assert "weather_data" in names
    assert names[-1] == "done"


def test_agent_error_emits_retry_error_and_done(monkeypatch):
    class FailingAgent:
        async def ainvoke(self, inputs, config=None):
            raise RuntimeError("model unavailable")

    async def builder(**kwargs):
        return FailingAgent(), FakeContext()

    store = ChatSessionStore(agent_builder=builder)
    monkeypatch.setattr(
        "travel_agent.api.sse_handler.load_settings",
        lambda _: SimpleNamespace(a2ui=SimpleNamespace(enabled=False)),
    )
    frames = _decode_frames(_collect(stream_chat(
        FakeRequest(),
        ChatStreamRequest(session_id="error-session", content="测试"),
        "unused.toml",
        store=store,
    )))
    names = [name for name, _ in frames]
    assert "retry" in names
    assert "error" in names
    assert frames[-1] == ("done", {"ok": False})


def test_disconnected_request_cancels_agent(monkeypatch):
    class SlowAgent:
        async def ainvoke(self, inputs, config=None):
            await asyncio.sleep(60)
            return {"messages": []}

    async def builder(**kwargs):
        return SlowAgent(), FakeContext()

    store = ChatSessionStore(agent_builder=builder)
    monkeypatch.setattr(
        "travel_agent.api.sse_handler.load_settings",
        lambda _: SimpleNamespace(a2ui=SimpleNamespace(enabled=False)),
    )

    async def run():
        stream = stream_chat(
            FakeRequest(disconnected=True),
            ChatStreamRequest(session_id="cancel-session", content="测试取消"),
            "unused.toml",
            store=store,
        )
        iterator = stream.__aiter__()
        assert (await iterator.__anext__()).startswith("event: session")
        assert (await iterator.__anext__()).startswith("event: memory_mode")
        try:
            await iterator.__anext__()
        except asyncio.CancelledError:
            return
        raise AssertionError("disconnect should cancel the SSE generator")

    asyncio.run(run())
