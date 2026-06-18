"""
MCP 会话生命周期管理器。

它在内存中缓存 ``session_id -> ArtifactStore`` 映射，并定期清理
过期或超过数量上限的会话目录。释放内存引用不会删除磁盘数据。
"""
from __future__ import annotations

import shutil
import time
import threading
import uuid
from pathlib import Path
from typing import Callable, Dict, Optional

from travel_agent.storage.agent_memory import ArtifactStore

try:
    from travel_agent.utils.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)


class SessionLifecycleManager:
    """
    管理每个 session 的 ArtifactStore，并清理过期数据。

    参数：
        artifacts_root: 会话 Artifact 的根目录；
        cache_root: MCP 服务端缓存目录；
        retention_days: 目录超过该天数后可被清理；
        max_sessions: 最多保留的会话目录数，超出后优先删除最旧目录；
        enable_cleanup: 是否启用清理逻辑。
    """

    def __init__(
        self,
        artifacts_root: str | Path,
        cache_root: str | Path,
        retention_days: int = 3,
        max_sessions: int = 256,
        enable_cleanup: bool = True,
    ) -> None:
        self.artifacts_root = Path(artifacts_root)
        self.cache_root = Path(cache_root)
        self.retention_days = retention_days
        self.max_sessions = max_sessions
        self.enable_cleanup = enable_cleanup

        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # 这里只缓存轻量 Store 对象；真实结果保存在文件系统。
        self._stores: Dict[str, ArtifactStore] = {}
        # MCP 请求可能并发获取/释放 Store，因此保护映射本身的修改。
        self._lock = threading.Lock()

    # ── 会话 Store 创建与缓存 ─────────────────────────────────

    def new_session(self) -> str:
        """生成新的随机会话 ID。"""
        return uuid.uuid4().hex

    def get_store(self, session_id: str) -> ArtifactStore:
        """返回指定会话的 ArtifactStore；不存在时按需创建。"""
        with self._lock:
            if session_id not in self._stores:
                self._stores[session_id] = ArtifactStore(
                    artifacts_dir=self.artifacts_root,
                    session_id=session_id,
                )
                logger.debug("[SessionMgr] created store for session %s", session_id)
            return self._stores[session_id]

    def release_session(self, session_id: str) -> None:
        """仅移除内存引用，不删除该会话已写入磁盘的 Artifact。"""
        with self._lock:
            self._stores.pop(session_id, None)
        logger.debug("[SessionMgr] released session %s", session_id)

    # ── 过期数据清理 ──────────────────────────────────────────

    def _safe_rmtree(self, path: Path) -> None:
        """删除目录；Windows 只读文件导致失败时先补写权限再重试。"""
        import os, stat as _stat

        def _on_error(func, p, exc):
            if not os.access(p, os.W_OK):
                os.chmod(p, _stat.S_IWUSR)
                func(p)
        if path.is_dir():
            shutil.rmtree(path, onerror=_on_error)
        else:
            path.unlink(missing_ok=True)

    def cleanup_expired(self, current_session_id: Optional[str] = None) -> None:
        """
        删除超过保留期的目录，并把总目录数压到 ``max_sessions`` 以内。

        ``current_session_id`` 用于保护仍在处理请求的会话不被本次清理删除。
        """
        if not self.enable_cleanup:
            return
        cutoff = time.time() - self.retention_days * 86_400

        try:
            all_dirs = [
                p for p in self.artifacts_root.iterdir()
                if p.is_dir() and p.name != current_session_id
            ]
        except FileNotFoundError:
            return

        expired = [p for p in all_dirs if p.stat().st_mtime < cutoff]
        for p in expired:
            logger.info("[SessionMgr] removing expired session dir: %s", p.name)
            self._safe_rmtree(p)
            with self._lock:
                self._stores.pop(p.name, None)

        remaining = [p for p in all_dirs if p not in expired]
        if len(remaining) > self.max_sessions:
            oldest = sorted(remaining, key=lambda p: p.stat().st_mtime)
            excess = oldest[: len(remaining) - self.max_sessions]
            for p in excess:
                logger.info("[SessionMgr] removing excess session dir: %s", p.name)
                self._safe_rmtree(p)
                with self._lock:
                    self._stores.pop(p.name, None)
