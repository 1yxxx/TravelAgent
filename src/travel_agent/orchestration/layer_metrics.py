"""
分层编排的追踪与度量模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LayerTrace:
    """单层执行追踪记录。"""
    layer: str
    attempt: int
    success: bool
    reason: str = ""
    tools_called: List[str] = field(default_factory=list)


@dataclass
class LayerMetrics:
    """分层编排的聚合度量。"""
    attempted_layers: int
    successful_layers: int
    layer_hit_rate: float
    total_attempts: int
    rollback_count: int
    rollback_rate: float
    per_layer_attempts: Dict[str, int] = field(default_factory=dict)
    per_layer_success: Dict[str, bool] = field(default_factory=dict)


def compute_layer_metrics(traces: List[LayerTrace]) -> LayerMetrics:
    """从层追踪记录中计算聚合度量。"""
    attempts = [t for t in traces if t.attempt > 0]
    per_layer_attempts: Dict[str, int] = {}
    per_layer_success: Dict[str, bool] = {}

    for t in attempts:
        per_layer_attempts[t.layer] = per_layer_attempts.get(t.layer, 0) + 1
        per_layer_success[t.layer] = per_layer_success.get(t.layer, False) or t.success

    attempted_layers = len(per_layer_attempts)
    successful_layers = sum(1 for ok in per_layer_success.values() if ok)
    total_attempts = len(attempts)
    rollback_count = sum(1 for t in attempts if not t.success)

    layer_hit_rate = (
        successful_layers / attempted_layers if attempted_layers > 0 else 0.0
    )
    rollback_rate = (
        rollback_count / total_attempts if total_attempts > 0 else 0.0
    )

    return LayerMetrics(
        attempted_layers=attempted_layers,
        successful_layers=successful_layers,
        layer_hit_rate=layer_hit_rate,
        total_attempts=total_attempts,
        rollback_count=rollback_count,
        rollback_rate=rollback_rate,
        per_layer_attempts=per_layer_attempts,
        per_layer_success=per_layer_success,
    )
