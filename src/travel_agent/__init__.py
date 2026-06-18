"""
TravelAgent — 基于 LangGraph ReAct 的多工具 AI 旅行规划 Agent。

包结构（重构后）:
- ``api/``:          Web & CLI 展示层（FastAPI、WebSocket、A2UI 桥接）
- ``agent/``:        Agent 装配层（工厂、上下文、DeepSeek 适配）
- ``tools/``:        核心工具层（search / planning / rendering / utility）
- ``mcp/``:          MCP 服务层（Server、工具注册、拦截器）
- ``storage/``:      三层记忆系统（L1 压缩 / L2 Artifact / L3 用户画像）
- ``orchestration/``: 分层编排（Policy / Validator / Metrics / Agent）
- ``skills/``:       Markdown Skills 热插拔
- ``utils/``:        工具模块（Prompts、Logging）
- ``config.py``:     Pydantic 配置模型
"""
