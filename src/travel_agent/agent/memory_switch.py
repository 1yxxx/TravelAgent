"""
记忆框架动态切换模块。

根据消息数量和 token 估算，在三套记忆模式间动态切换：
- full_context: 原始上下文 + L2 + L3
- compressed_context: L1 压缩 + L2 + L3
- profile_only: L1 压缩 + L3（关闭 L2 注入）
"""

from __future__ import annotations

from typing import Tuple


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
    三套记忆框架动态切换。

    Returns:
        (mode, reason)
    """
    if not enabled:
        return "full_context", "memory_switch.disabled"

    if message_count >= hard_message_threshold or token_estimate >= hard_token_threshold:
        return "profile_only", "hard-threshold"

    if message_count >= soft_message_threshold or token_estimate >= soft_token_threshold:
        return "compressed_context", "soft-threshold"

    return "full_context", "normal"
