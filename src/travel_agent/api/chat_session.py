"""
HTTP 聊天会话的内存状态管理。

SSE 是一次请求对应一条单向响应流，不像 WebSocket 可以把状态挂在连接上。
因此这里按 ``session_id`` 保存 Agent、运行时上下文和 LangChain messages。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from langchain_core.messages import BaseMessage

from travel_agent.agent.factory import build_agent


AgentBuilder = Callable[..., Awaitable[tuple[Any, Any]]]


@dataclass
class ChatSession:
    """一个浏览器会话对应的后端运行状态。"""

    session_id: str
    agent: Any = None
    context: Any = None
    messages: List[BaseMessage] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_access: float = field(default_factory=time.monotonic)


class ChatSessionStore:
    """带会话锁和惰性过期清理的进程内会话仓库。

    该实现适用于当前 Docker 单 worker 部署。若未来扩展为多 worker，需要把
    messages 移到 Redis/数据库，并将同一 session 的请求路由到同一状态节点。
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 2 * 60 * 60,
        agent_builder: AgentBuilder = build_agent,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.agent_builder = agent_builder
        self._sessions: Dict[str, ChatSession] = {}
        self._store_lock = asyncio.Lock()

    async def get_or_create(self, session_id: str) -> ChatSession:
        """取得会话；访问时顺便清理已过期且未执行请求的会话。"""
        now = time.monotonic()
        async with self._store_lock:
            expired = [
                key
                for key, value in self._sessions.items()
                if not value.lock.locked()
                and now - value.last_access > self.ttl_seconds
            ]
            for key in expired:
                self._sessions.pop(key, None)

            session = self._sessions.get(session_id)
            if session is None:
                session = ChatSession(session_id=session_id)
                self._sessions[session_id] = session
            session.last_access = now
            return session

    async def ensure_initialized(self, session: ChatSession, cfg: Any) -> None:
        """在会话锁内惰性创建 Agent，避免无请求时提前占用 MCP/LLM 资源。"""
        if session.agent is not None and session.context is not None:
            return
        session.agent, session.context = await self.agent_builder(
            cfg=cfg,
            session_id=session.session_id,
            lang="zh",
        )

    async def remove(self, session_id: str) -> Optional[ChatSession]:
        async with self._store_lock:
            return self._sessions.pop(session_id, None)

    async def size(self) -> int:
        async with self._store_lock:
            return len(self._sessions)

