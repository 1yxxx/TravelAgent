"""
旅行工具 MCP Server 的创建与独立启动入口。

Agent 通过 ``MultiServerMCPClient`` 连接本服务。每次工具请求都携带
``X-Travel-Session-Id``，工具注册层据此获取对应会话的 ArtifactStore，
从而隔离不同会话产生的搜索、天气和行程结果。

本模块只负责创建和运行 MCP Server，具体工具定义位于
``mcp/register_tools.py``。

可独立运行：``python -m travel_agent.mcp.server``；
Web 模式下则由 ``api/server.py`` 的 lifespan 在后台启动。
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from travel_agent.config import Settings, load_settings, default_config_path
from travel_agent.mcp import register_tools
from travel_agent.storage.session_manager import SessionLifecycleManager

try:
    from travel_agent.utils.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)


def create_server(cfg: Settings) -> FastMCP:
    """
    创建并配置 FastMCP 实例，但不在此函数中启动网络监听。

    调用方可以执行 ``server.run(...)``，也可以获取 ASGI app 后交给
    Uvicorn 托管。
    """

    @asynccontextmanager
    async def session_lifespan(server: FastMCP) -> AsyncIterator[SessionLifecycleManager]:
        # lifespan 的返回值会作为 MCP 请求上下文共享给各工具，
        # register_tools 通过它取得按 session_id 隔离的 ArtifactStore。
        logger.info("[MCP] starting session lifecycle manager …")
        mgr = SessionLifecycleManager(
            artifacts_root=cfg.project.outputs_dir,
            cache_root=cfg.mcp_server.server_cache_dir,
            enable_cleanup=True,
        )
        try:
            yield mgr
        finally:
            logger.info("[MCP] cleaning up expired sessions …")
            mgr.cleanup_expired(current_session_id=None)

    server = FastMCP(
        name=cfg.mcp_server.server_name,
        stateless_http=cfg.mcp_server.stateless_http,
        json_response=cfg.mcp_server.json_response,
        lifespan=session_lifespan,
    )

    # 工具在 Server 创建阶段一次性注册；运行期间只处理调用和会话数据。
    register_tools.register(server, cfg)
    logger.info("[MCP] server '%s' created with %d tool(s)", cfg.mcp_server.server_name, len(server._tool_manager._tools))
    return server


def main() -> None:
    """独立启动 MCP Server，主要用于调试或供外部 MCP Client 连接。"""
    cfg = load_settings(default_config_path())
    server = create_server(cfg)
    server.settings.host = cfg.mcp_server.connect_host
    server.settings.port = cfg.mcp_server.port
    server.run(transport=cfg.mcp_server.server_transport)


if __name__ == "__main__":
    main()
