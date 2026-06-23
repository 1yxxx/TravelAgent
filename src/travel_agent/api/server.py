"""
FastAPI Web 服务入口。

负责：
1. FastAPI 与内置 MCP Server 的启动/关闭生命周期；
2. 托管静态前端，并提供 SSE 聊天、上传、导出等 HTTP 接口。
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# 项目根目录（travel/）
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(ROOT_DIR, "config.toml")):
    parent = os.path.dirname(ROOT_DIR)
    if parent == ROOT_DIR:
        raise RuntimeError("Cannot find project root (config.toml)")
    ROOT_DIR = parent

CONFIG_PATH = os.path.join(ROOT_DIR, "config.toml")
WEB_DIR = os.path.join(ROOT_DIR, "web")
STATIC_DIR = os.path.join(WEB_DIR, "static")
INDEX_HTML = os.path.join(WEB_DIR, "index.html")

from travel_agent.config import load_settings
from travel_agent.utils.logging import logger
from travel_agent.api.file_routes import register_file_routes
from travel_agent.api.sse_handler import (
    ChatStreamRequest,
    create_sse_response,
    stream_chat,
)

# ── MCP Server 后台任务 ────────────────────────────────────────────────────────
_mcp_server_task: Optional[asyncio.Task] = None


async def _run_mcp_server(cfg) -> None:
    """在后台 asyncio task 中启动 MCP Server（streamable-http 模式）。"""
    from travel_agent.mcp.server import create_server
    import uvicorn

    mcp_server = create_server(cfg)
    mcp_asgi = mcp_server.streamable_http_app()
    uv_cfg = uvicorn.Config(
        app=mcp_asgi,
        host=cfg.mcp_server.connect_host,
        port=cfg.mcp_server.port,
        log_level="warning",
        timeout_keep_alive=300,
    )
    server = uvicorn.Server(uv_cfg)
    logger.info(
        "[MCP] Server starting on %s:%s%s",
        cfg.mcp_server.connect_host, cfg.mcp_server.port, cfg.mcp_server.path,
    )
    await server.serve()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时拉起 MCP Server；关闭时取消 task。"""
    cfg = load_settings(CONFIG_PATH)
    global _mcp_server_task
    _mcp_server_task = asyncio.create_task(_run_mcp_server(cfg))
    await asyncio.sleep(1.5)
    logger.info("[FastAPI] MCP Server task started")
    try:
        yield
    finally:
        if _mcp_server_task and not _mcp_server_task.done():
            _mcp_server_task.cancel()
            try:
                await _mcp_server_task
            except asyncio.CancelledError:
                pass
        logger.info("[FastAPI] MCP Server task stopped")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。"""
    app = FastAPI(title="Travel Smart Assistant", lifespan=lifespan)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> JSONResponse:
        if _mcp_server_task is None or _mcp_server_task.done():
            return JSONResponse(
                content={"status": "error", "component": "mcp"}, status_code=503
            )
        return JSONResponse(content={"status": "ok"})

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/apple-touch-icon.png", include_in_schema=False)
    async def apple_touch_icon() -> Response:
        return Response(status_code=204)

    @app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
    async def apple_touch_icon_precomposed() -> Response:
        return Response(status_code=204)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        cfg = load_settings(CONFIG_PATH)
        jsapi_key = (cfg.map.jsapi_key or "").strip()
        html = Path(INDEX_HTML).read_text(encoding="utf-8")
        loader_placeholder = (
            '<script type="text/javascript" id="amap-loader"></script>'
        )
        # Web 服务 Key 与 JS API Key 是两类凭据。未配置 JS API Key 时，
        # 保留聊天、Agent、MCP、天气等主链路，并让前端显示清晰的地图降级提示。
        if jsapi_key and not jsapi_key.upper().startswith("YOUR_"):
            map_loader = (
                '<script type="text/javascript" '
                f'src="https://webapi.amap.com/maps?v=2.0&key={jsapi_key}">'
                "</script>"
            )
        else:
            map_loader = (
                "<script>"
                "window.TRAVEL_MAP_ENABLED=false;"
                "window.TRAVEL_MAP_DISABLED_REASON='未配置高德 JS API Key';"
                "</script>"
            )
        html = html.replace(loader_placeholder, map_loader)
        return html

    @app.post("/api/chat/stream")
    async def chat_stream(request: Request, payload: ChatStreamRequest):
        return create_sse_response(stream_chat(request, payload, CONFIG_PATH))

    register_file_routes(app, CONFIG_PATH)

    return app


def main():
    """启动 FastAPI 服务器。"""
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
