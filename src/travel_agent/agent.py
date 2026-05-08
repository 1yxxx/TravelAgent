from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple
import time
import re

from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
import json as _json
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from travel_agent.config import Settings
from travel_agent.storage.memory_compressor import MemoryCompressor
from travel_agent.storage.user_profile import UserProfileStore
from travel_agent.storage.agent_memory import ArtifactStore


def _flatten_content(content) -> str:
    """把 list/dict 类型的 message content 拍平为字符串，DeepSeek 只接受 string。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or _json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return _json.dumps(content, ensure_ascii=False)


def _estimate_tokens(messages: List[BaseMessage]) -> int:
    """粗略估算 token，用于上下文爆炸保护。"""
    total = 0
    for msg in messages:
        content = _flatten_content(getattr(msg, "content", "") or "")
        for ch in content:
            total += 2 if "\u4e00" <= ch <= "\u9fff" else 1
    return total // 4


def choose_memory_framework(
    *,
    enabled: bool,
    message_count: int,
    token_estimate: int,
    soft_message_threshold: int,
    hard_message_threshold: int,
    soft_token_threshold: int,
    hard_token_threshold: int,
) -> Tuple[str, str]:
    """
    三套记忆框架动态切换：
    - full_context: 原始上下文 + L2 + L3
    - compressed_context: L1 压缩 + L2 + L3
    - profile_only: L1 压缩 + L3（关闭 L2 注入）
    """
    if not enabled:
        return "full_context", "memory_switch.disabled"

    if message_count >= hard_message_threshold or token_estimate >= hard_token_threshold:
        return "profile_only", "hard-threshold"

    if message_count >= soft_message_threshold or token_estimate >= soft_token_threshold:
        return "compressed_context", "soft-threshold"

    return "full_context", "normal"


_COMMON_CITY_HINTS = [
    "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "成都", "重庆", "西安", "武汉", "长沙", "青岛", "厦门", "三亚",
    "昆明", "大理", "丽江", "贵阳", "哈尔滨", "长春", "沈阳", "天津", "福州", "南昌", "郑州", "济南",
]


def _has_travel_intent(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in (
        "旅行", "旅游", "行程", "攻略", "出游", "自驾", "自由行", "规划", "路线", "景点", "酒店", "机票",
        "trip", "travel", "itinerary", "route",
    ))


def _has_destination(text: str) -> bool:
    if any(city in text for city in _COMMON_CITY_HINTS):
        return True
    # 兜底：匹配“去XX”这类短语
    if re.search(r"去[\u4e00-\u9fa5]{2,8}", text):
        return True
    return False


def _has_duration(text: str) -> bool:
    t = text.lower()
    if "周末" in text or "假期" in text:
        return True
    return bool(re.search(r"\d+\s*(天|日|晚)", t))


class DeepSeekChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI 的子类，在发送请求前将所有消息的 content 强制转为 string，
    解决 DeepSeek API 不接受 list 类型 content 的问题。
    同时回注 reasoning_content（DeepSeek thinking 模式要求回传）。
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # 从原始消息中收集 reasoning_content（父类 _convert_message_to_dict 会丢弃它）
        reasoning_list: list[str] = []
        if isinstance(input_, list):
            for m in input_:
                rc = getattr(m, "reasoning_content", None)
                if rc:
                    reasoning_list.append(rc)

        fixed = []
        rc_idx = 0
        for msg in payload.get("messages", []):
            content = msg.get("content")
            if content is not None and not isinstance(content, str):
                msg = dict(msg)
                msg["content"] = _flatten_content(content)
            # 将 reasoning_content 回注到对应的 assistant 消息
            if msg.get("role") == "assistant" and rc_idx < len(reasoning_list):
                if not isinstance(msg, dict):
                    msg = dict(msg)
                msg["reasoning_content"] = reasoning_list[rc_idx]
                rc_idx += 1
            fixed.append(msg)
        payload["messages"] = fixed
        return payload


from travel_agent.nodes.node_manager import NodeManager
from travel_agent.orchestration.layered_agent import LayerPolicy, LayeredTravelAgent
from travel_agent.utils.prompts import get_system_prompt
from travel_agent.utils.logging import logger
from travel_agent.skills.skills_io import load_skills

@dataclass
class ClientContext:
    cfg: Settings
    session_id: str
    node_manager: NodeManager
    lang: str = "zh"
    mcp_client: Any = field(default=None)          # MultiServerMCPClient，供外部关闭
    memory_compressor: Any = field(default=None)   # L1: MemoryCompressor
    user_profile: Any = field(default=None)        # L3: UserProfileStore
    artifact_store: Optional[ArtifactStore] = field(default=None)
    _base_system_prompt: str = field(default="", repr=False)

    def precheck_user_request(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """前置业务校验：意图、目的地、天数。"""
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        if not humans:
            return {"ok": True}

        text = _flatten_content(getattr(humans[-1], "content", "") or "")
        if len(text.strip()) < 2:
            return {
                "ok": False,
                "reason": "query_too_short",
                "suggestion": "请补充目的地和出行天数，例如：成都3天亲子游，预算5000。",
            }

        if not _has_travel_intent(text):
            return {
                "ok": False,
                "reason": "intent_unclear",
                "suggestion": "我可以帮你规划旅行，请告诉我目的地、天数和预算。",
            }

        if not _has_destination(text):
            return {
                "ok": False,
                "reason": "missing_destination",
                "suggestion": "请先告诉我目的地城市，例如：杭州、成都、青岛。",
            }

        if not _has_duration(text):
            return {
                "ok": False,
                "reason": "missing_duration",
                "suggestion": "请补充出行天数，例如：2天、3天或周末两天。",
            }

        return {"ok": True}

    async def prepare_messages_for_invoke(
        self,
        messages: List[BaseMessage],
    ) -> tuple[List[BaseMessage], Dict[str, Any]]:
        """
        统一入口：
        1) 根据消息规模动态切换记忆框架
        2) 触发 L1 压缩（必要时）
        3) 注入动态 system prompt（L2/L3 按策略启用）
        """
        working = [m for m in messages if not isinstance(m, SystemMessage)]
        token_est = _estimate_tokens(working)
        mode, reason = choose_memory_framework(
            enabled=self.cfg.memory_switch.enabled,
            message_count=len(working),
            token_estimate=token_est,
            soft_message_threshold=self.cfg.memory_switch.soft_message_threshold,
            hard_message_threshold=self.cfg.memory_switch.hard_message_threshold,
            soft_token_threshold=self.cfg.memory_switch.soft_token_threshold,
            hard_token_threshold=self.cfg.memory_switch.hard_token_threshold,
        )

        # L3 偏好提取（轻量规则，无需 LLM）
        if self.user_profile is not None:
            try:
                self.user_profile.extract_preferences_from_messages(working, lang=self.lang)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[Memory] preference extraction skipped: %s", exc)

        # L1 压缩
        if mode in {"compressed_context", "profile_only"} and self.memory_compressor is not None:
            try:
                working = await self.memory_compressor.maybe_compress(working)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Memory] L1 compression failed: %s", exc)

        # L2 注入策略：profile_only 关闭 L2，仅保留 L3
        store_for_prompt = None if mode == "profile_only" else self.artifact_store
        dynamic_prompt = self.build_dynamic_system_prompt(store=store_for_prompt)
        invoke_messages: List[BaseMessage] = [SystemMessage(content=dynamic_prompt)] + working

        return invoke_messages, {
            "mode": mode,
            "reason": reason,
            "message_count": len(working),
            "token_estimate": token_est,
        }

    def build_quality_feedback(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """从 layer_trace 中提炼双向反馈（成功/失败 + 失败原因）。"""
        trace = result.get("layer_trace")
        if not isinstance(trace, list):
            return {"enabled": False}

        failures: List[Dict[str, Any]] = []
        for item in trace:
            if not isinstance(item, dict):
                continue
            if item.get("success") is False:
                failures.append(
                    {
                        "layer": item.get("layer", ""),
                        "attempt": item.get("attempt", 0),
                        "reason": item.get("reason", ""),
                    }
                )

        return {
            "enabled": True,
            "status": "ok" if not failures else "needs_retry",
            "failures": failures,
            "failure_count": len(failures),
        }

    def build_dynamic_system_prompt(
        self,
        store=None,          # ArtifactStore | None，用于 L2 context snapshot
    ) -> str:
        """
        动态拼装 system prompt，融合三层记忆：

        ┌─────────────────────────────────────────┐
        │ 原始 system prompt（指令 + 工具说明）      │
        ├─────────────────────────────────────────┤
        │ L3：用户偏好 + 历史 session 摘要          │
        ├─────────────────────────────────────────┤
        │ L2：本 session 已收集工具结果（snapshot）  │
        └─────────────────────────────────────────┘
        """
        parts: list[str] = [self._base_system_prompt]

        # L3：用户偏好画像
        if self.user_profile is not None:
            profile_text = self.user_profile.build_profile_prompt(lang=self.lang)
            if profile_text:
                parts.append("\n\n" + profile_text)

        # L2：本 session 工具结果快照
        if store is not None:
            snapshot_text = store.build_context_prompt(lang=self.lang)
            if snapshot_text:
                parts.append("\n\n" + snapshot_text)

        return "".join(parts)

    def persist_layer_metrics(self, result: Dict[str, Any]) -> Optional[str]:
        """
        将分层编排指标写入 session artifacts，供离线评估脚本聚合。

        Returns:
            artifact_id or None
        """
        if self.artifact_store is None:
            return None

        metrics = result.get("layer_metrics")
        if not isinstance(metrics, dict):
            return None

        trace = result.get("layer_trace")
        payload = {
            "session_id": self.session_id,
            "created_at": time.time(),
            "layer_metrics": metrics,
            "layer_trace": trace if isinstance(trace, list) else [],
        }
        summary = (
            f"分层指标: hit={metrics.get('layer_hit_rate', 0):.3f} "
            f"rollback={metrics.get('rollback_rate', 0):.3f}"
        )
        meta = self.artifact_store.save_result(
            node_id="layer_metrics",
            payload=payload,
            summary=summary,
        )
        return meta.artifact_id


def _build_llm(cfg: Settings) -> BaseChatModel:
    # DeepSeek API 不接受 list 类型的 content，需要用子类拍平；
    # 其他兼容 OpenAI 格式的服务（智谱、通义等）直接用标准 ChatOpenAI。
    cls = DeepSeekChatOpenAI if "deepseek" in cfg.llm.base_url.lower() else ChatOpenAI
    return cls(
        model=cfg.llm.model,
        base_url=cfg.llm.base_url,
        api_key=cfg.llm.api_key,
        timeout=cfg.llm.timeout,
    )


def _collect_tools(cfg: Settings) -> List[BaseTool]:
    """
    已废弃：工具现在统一从 MCP Server 获取。
    保留此函数仅供 fallback / 测试使用。
    """
    from travel_agent.nodes.core_nodes.search_poi import search_poi_tool
    from travel_agent.nodes.core_nodes.plan_itinerary import plan_itinerary_tool
    from travel_agent.nodes.core_nodes.estimate_budget import estimate_budget_tool
    from travel_agent.nodes.core_nodes.recommend_transport import recommend_transport_tool
    from travel_agent.nodes.core_nodes.check_weather import check_weather_tool
    from travel_agent.nodes.core_nodes.json_tools import validate_json_tool, fix_json_tool
    from travel_agent.nodes.core_nodes.search_hotel import search_hotel_tool
    from travel_agent.nodes.core_nodes.search_restaurant import search_restaurant_tool
    from travel_agent.nodes.core_nodes.plan_route import plan_route_tool
    from travel_agent.nodes.core_nodes.format_itinerary import format_itinerary_tool
    from travel_agent.nodes.core_nodes.render_map import render_map_pois_tool, render_map_route_tool
    from travel_agent.nodes.core_nodes.render_itinerary import render_itinerary_tool
    from travel_agent.nodes.core_nodes.smart_plan_itinerary import smart_plan_itinerary_tool
    from travel_agent.nodes.core_nodes.request_travel_info import request_travel_info_tool
    return [
        search_poi_tool, plan_itinerary_tool, estimate_budget_tool,
        recommend_transport_tool, check_weather_tool, validate_json_tool, fix_json_tool,
        search_hotel_tool, search_restaurant_tool, plan_route_tool, format_itinerary_tool,
        render_map_pois_tool, render_map_route_tool, render_itinerary_tool,
        smart_plan_itinerary_tool, request_travel_info_tool,
    ]


async def build_agent(cfg: Settings, session_id: str, *, lang: str = "zh"):
    """
    构建旅行助手 Agent：
    - 通过 MultiServerMCPClient 连接本地 MCP Server，获取所有工具
    - 动态从 .storyline/skills 目录加载 skills
    - 使用 LangGraph create_react_agent 构建具备函数调用能力的对话 Agent
    - 初始化三层记忆组件（L1 MemoryCompressor / L3 UserProfileStore）
    """
    from pathlib import Path

    llm = _build_llm(cfg)

    # ── 连接 MCP Server，获取工具 ──────────────────────────────────────
    mcp_url = (
        f"{cfg.mcp_server.url_scheme}://{cfg.mcp_server.connect_host}"
        f":{cfg.mcp_server.port}{cfg.mcp_server.path}"
    )
    client = MultiServerMCPClient(
        connections={
            cfg.mcp_server.server_name: {
                "transport": "streamable_http",
                "url": mcp_url,
                "timeout": timedelta(seconds=cfg.mcp_server.timeout),
                "headers": {"X-Travel-Session-Id": session_id},
            }
        }
    )
    tools: List[BaseTool] = await client.get_tools()
    logger.info("[Agent] fetched %d tools from MCP Server at %s", len(tools), mcp_url)

    # ── 加载 Markdown Skills ───────────────────────────────────────────
    _travel_root = Path(__file__).resolve().parent.parent.parent  # travel/
    skill_dir = str(_travel_root / ".storyline" / "skills")
    try:
        skills_tools = await load_skills(skill_dir=skill_dir)
        tools = tools + skills_tools
        logger.info("[Agent] loaded %d skills from %s", len(skills_tools), skill_dir)
    except Exception as exc:
        logger.warning("[Agent] Skills load failed (non-fatal): %s", exc)

    system_prompt = get_system_prompt(lang=lang)
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    # ── 三层记忆：L1 MemoryCompressor ────────────────────────────────
    _outputs_root = Path(cfg.project.outputs_dir)
    session_dir = _outputs_root / session_id
    compressor = MemoryCompressor(
        llm=llm,
        session_dir=session_dir,
        lang=lang,
    )

    # ── 三层记忆：L3 UserProfileStore ────────────────────────────────
    user_profile = UserProfileStore(
        data_dir=cfg.project.data_dir,
        user_id="default",
    )

    artifact_store = ArtifactStore(
        artifacts_dir=cfg.project.outputs_dir,
        session_id=session_id,
    )

    node_manager = NodeManager(tools=tools)

    if cfg.orchestration.enabled:
        policy = LayerPolicy(
            enabled=True,
            max_retries_per_layer=cfg.orchestration.max_retries_per_layer,
            strict_validation=cfg.orchestration.strict_validation,
        )
        agent = LayeredTravelAgent(
            llm=llm,
            tools_by_layer=node_manager.grouped_tools(),
            base_system_prompt=system_prompt,
            policy=policy,
            runtime_tool_selector=node_manager.grouped_tools_for_messages,
        )
        logger.info(
            "[Agent] layered orchestration enabled: retries=%s strict=%s",
            policy.max_retries_per_layer,
            policy.strict_validation,
        )

    context = ClientContext(
        cfg=cfg,
        session_id=session_id,
        node_manager=node_manager,
        lang=lang,
        mcp_client=client,
        memory_compressor=compressor,
        user_profile=user_profile,
        artifact_store=artifact_store,
        _base_system_prompt=system_prompt,
    )

    logger.info("[Agent] built for session %s with %d tools (via MCP)", session_id, len(tools))
    return agent, context

