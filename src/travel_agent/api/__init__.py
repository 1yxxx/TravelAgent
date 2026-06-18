"""
Web & CLI 展示层。

子模块：
- ``server``:          FastAPI Web 服务入口（需安装依赖）
- ``websocket_handler``: WebSocket 聊天端点
- ``a2ui_bridge``:    A2UI 卡片生成与发送
- ``map_extractor``:  地图/天气 JSON 提取
- ``message_utils``:  消息序列化与清理
- ``html_render``:    行程 HTML 渲染（PDF 导出）
- ``file_routes``:    文件上传/导出路由
- ``cli``:            CLI 交互入口
"""

from travel_agent.api.a2ui_bridge import build_form_card_payload, extract_place_cards
from travel_agent.api.map_extractor import extract_map_blocks, extract_weather_block
from travel_agent.api.message_utils import normalize_content, clean_messages_for_next_turn

__all__ = [
    "build_form_card_payload",
    "extract_place_cards",
    "extract_map_blocks",
    "extract_weather_block",
    "normalize_content",
    "clean_messages_for_next_turn",
]
