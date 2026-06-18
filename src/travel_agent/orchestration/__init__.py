"""
旅行 Agent 分层编排包。

提供五层编排（requirement→research→planning→risk→render）的完整工具链：
- ``LayerPolicy``: 编排策略参数
- ``LayerValidator``: 层执行最小校验
- ``LayerTrace`` / ``LayerMetrics``: 追踪与度量
- ``LayeredTravelAgent``: 五层编排包装器
- ``LAYER_ORDER``: 五层顺序常量
- ``infer_scenario_tags``: 场景标签识别（共享工具）
"""

from travel_agent.orchestration.layer_policy import LAYER_ORDER, LayerPolicy
from travel_agent.orchestration.layer_validator import LayerValidator
from travel_agent.orchestration.layer_metrics import LayerTrace, LayerMetrics, compute_layer_metrics
from travel_agent.orchestration.layered_agent import LayeredTravelAgent
from travel_agent.orchestration.scenario_tags import infer_scenario_tags, _extract_text_from_messages

__all__ = [
    "LAYER_ORDER",
    "LayerPolicy",
    "LayerValidator",
    "LayerTrace",
    "LayerMetrics",
    "compute_layer_metrics",
    "LayeredTravelAgent",
    "infer_scenario_tags",
]
