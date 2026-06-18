"""
分层编排的策略配置与常量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

# 五层编排顺序（唯一权威定义）
LAYER_ORDER: List[str] = [
    "requirement",
    "research",
    "planning",
    "risk",
    "render",
]


@dataclass
class LayerPolicy:
    """分层编排策略参数。"""
    enabled: bool = False
    max_retries_per_layer: int = 1
    strict_validation: bool = False
