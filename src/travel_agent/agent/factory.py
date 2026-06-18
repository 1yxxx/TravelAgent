"""
Agent 装配工厂。

``build_agent()`` 类似 Java 的 Application Configuration/Factory：
负责创建 LLM、通过 MCP 获取工具、加载 Skill、构建 ReAct Agent，
并为当前会话装配 L1/L2/L3 记忆组件。
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from travel_agent.config import Settings
from travel_agent.agent.deepseek_adapter import DeepSeekChatOpenAI
from travel_agent.agent.context import ClientContext
from travel_agent.agent.node_manager import NodeManager
from travel_agent.storage.memory_compressor import MemoryCompressor
from travel_agent.storage.user_profile import UserProfileStore
from travel_agent.storage.agent_memory import ArtifactStore
from travel_agent.orchestration import LayerPolicy, LayeredTravelAgent
from travel_agent.utils.prompts import get_system_prompt
from travel_agent.utils.logging import logger
from travel_agent.skills.skills_io import load_skills


def _build_llm(cfg: Settings) -> BaseChatModel:
    """根据兼容接口类型创建聊天模型，并对 DeepSeek 做协议适配。"""
    cls = DeepSeekChatOpenAI if "deepseek" in cfg.llm.base_url.lower() else ChatOpenAI
    return cls(
        model=cfg.llm.model,
        base_url=cfg.llm.base_url,
        api_key=cfg.llm.api_key,
        timeout=cfg.llm.timeout,
    )


def _collect_tools(cfg: Settings) -> List[BaseTool]:
    """已废弃：工具统一从 MCP Server 获取。保留仅供 fallback / 测试使用。"""
    from travel_agent.tools.search.search_poi import search_poi_tool
    from travel_agent.tools.planning.plan_itinerary import plan_itinerary_tool
    from travel_agent.tools.planning.estimate_budget import estimate_budget_tool
    from travel_agent.tools.planning.recommend_transport import recommend_transport_tool
    from travel_agent.tools.search.check_weather import check_weather_tool
    from travel_agent.tools.utility.json_tools import validate_json_tool, fix_json_tool
    from travel_agent.tools.search.search_hotel import search_hotel_tool
    from travel_agent.tools.search.search_restaurant import search_restaurant_tool
    from travel_agent.tools.planning.plan_route import plan_route_tool
    from travel_agent.tools.planning.format_itinerary import format_itinerary_tool
    from travel_agent.tools.rendering.render_map import render_map_pois_tool, render_map_route_tool
    from travel_agent.tools.rendering.render_itinerary import render_itinerary_tool
    from travel_agent.tools.planning.smart_plan_itinerary import smart_plan_itinerary_tool
    from travel_agent.tools.utility.request_travel_info import request_travel_info_tool
    return [
        search_poi_tool, plan_itinerary_tool, estimate_budget_tool,
        recommend_transport_tool, check_weather_tool, validate_json_tool, fix_json_tool,
        search_hotel_tool, search_restaurant_tool, plan_route_tool, format_itinerary_tool,
        render_map_pois_tool, render_map_route_tool, render_itinerary_tool,
        smart_plan_itinerary_tool, request_travel_info_tool,
    ]


async def build_agent(cfg: Settings, session_id: str, *, lang: str = "zh"):
    """构建旅行助手 Agent。

    通过 MultiServerMCPClient 连接本地 MCP Server 获取工具，
    动态加载 Skills，创建 LangGraph ReAct Agent，
    并初始化三层记忆组件（L1/L2/L3）。
    """
    # LLM 实例
    llm = _build_llm(cfg)

    # ── 连接 MCP Server，获取工具 ──
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

    # ── 加载 Markdown Skills ──
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

    # ── 三层记忆：L1 MemoryCompressor ──
    _outputs_root = Path(cfg.project.outputs_dir)
    session_dir = _outputs_root / session_id
    compressor = MemoryCompressor(
        llm=llm,
        session_dir=session_dir,
        lang=lang,
    )

    # ── 三层记忆：L3 UserProfileStore ──
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
            policy.max_retries_per_layer, policy.strict_validation,
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
