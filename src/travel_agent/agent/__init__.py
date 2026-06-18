"""
Agent 装配包。

子模块：
- ``factory``:          Agent 工厂函数 (需要安装 langchain_mcp_adapters)
- ``context``:          会话运行时上下文 ClientContext
- ``deepseek_adapter``: DeepSeek API 协议适配
- ``precheck``:         前置校验
- ``memory_switch``:    记忆框架动态切换
- ``node_manager``:     工具元数据管理

用法:
    from travel_agent.agent.factory import build_agent  # 生产环境
    from travel_agent.agent import ClientContext, NodeManager  # 轻量导入
"""

from travel_agent.agent.context import ClientContext
from travel_agent.agent.deepseek_adapter import DeepSeekChatOpenAI, _flatten_content, _estimate_tokens
from travel_agent.agent.node_manager import NodeManager
from travel_agent.agent.precheck import precheck_user_request
from travel_agent.agent.memory_switch import choose_memory_framework

__all__ = [
    "ClientContext",
    "DeepSeekChatOpenAI",
    "NodeManager",
    "precheck_user_request",
    "choose_memory_framework",
    "_flatten_content",
    "_estimate_tokens",
]
