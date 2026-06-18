"""
Agent 会话运行时上下文。

``ClientContext`` 保存不直接属于 LangGraph messages 的运行时依赖，
例如 ArtifactStore、用户画像和记忆压缩器。同时提供动态 system prompt 构建、
质量反馈和分层指标持久化等能力。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from travel_agent.config import Settings
from travel_agent.storage.agent_memory import ArtifactStore
from travel_agent.agent.deepseek_adapter import _flatten_content, _estimate_tokens
from travel_agent.agent.memory_switch import choose_memory_framework
from travel_agent.agent.precheck import precheck_user_request


@dataclass
class ClientContext:
    """单个 Agent 会话的运行时上下文，不在不同用户连接之间共享。"""

    cfg: Settings
    session_id: str
    node_manager: Any  # NodeManager
    lang: str = "zh"
    mcp_client: Any = field(default=None)          # MultiServerMCPClient
    memory_compressor: Any = field(default=None)   # L1: MemoryCompressor
    user_profile: Any = field(default=None)        # L3: UserProfileStore
    artifact_store: Optional[ArtifactStore] = field(default=None)
    _base_system_prompt: str = field(default="", repr=False)

    def precheck_user_request(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """前置业务校验：意图、目的地、天数。

        Returns:
            dict with ok, reason, suggestion fields.
        """
        return precheck_user_request(messages)

    async def prepare_messages_for_invoke(
        self,
        messages: List[BaseMessage],
    ) -> tuple[List[BaseMessage], Dict[str, Any]]:
        """统一入口：记忆框架切换 → L1压缩 → 动态 system prompt。

        Returns:
            (invoke_messages, memory_meta)
        """
        from travel_agent.utils.logging import logger

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

        # L2 注入策略
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
                failures.append({
                    "layer": item.get("layer", ""),
                    "attempt": item.get("attempt", 0),
                    "reason": item.get("reason", ""),
                })

        return {
            "enabled": True,
            "status": "ok" if not failures else "needs_retry",
            "failures": failures,
            "failure_count": len(failures),
        }

    def build_dynamic_system_prompt(
        self,
        store=None,  # ArtifactStore | None
    ) -> str:
        """动态拼装 system prompt，融合 L2/L3 记忆层。"""
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
        """将分层编排指标写入 session artifacts。

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
