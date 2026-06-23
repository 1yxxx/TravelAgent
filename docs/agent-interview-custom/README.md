# TravelAgent Agent 面试题项目化回答

> 题目来源：[Zero2Agent · Agent 面试通关](https://onefly.top/zero2Agent/learn-agent-interview/)  
> 抓取时间：2026-06-19  
> 抓取规则：专题页内所有以 `Q：` 开头的题目，以及正文中的 `追问：`。  
> 总计：433 道主问题与追问，14 个实际题目专题；“各公司面试偏好”页是索引页，没有独立 Q 题。

为避免机械复制原站内容，文档题干按原页面顺序进行语义等价转述；序号、专题归属和题目数量保持一一对应，章节顶部提供原页面链接用于核对。

这套回答以当前 `TravelAgent` 工作区代码为事实依据。回答分三类：

1. **项目已经实现**：说明真实代码路径、执行流程和工程权衡。
2. **项目部分实现**：说明当前能力、缺口和下一步生产化方案。
3. **项目尚未实现**：明确说明没有实现，不伪造经验；随后给出如果在本项目落地的具体方案。

## 项目事实基线

| 主题 | 当前项目实现 |
|---|---|
| Agent | `agent/factory.py` 使用 LangGraph `create_react_agent`；配置开启时替换为 `LayeredTravelAgent` |
| 分层编排 | `requirement → research → planning → risk → render`，每层有工具白名单、校验、checkpoint、回滚与指标 |
| 工具 | FastMCP 注册 17 个工具，Agent 通过 `MultiServerMCPClient` 获取；按 `X-Travel-Session-Id` 隔离 |
| Skills | `.storyline/skills/` 下 4 个 Markdown Skill，经 SkillKit 转为 LangChain Tool |
| 记忆 | L1 对话摘要压缩；L2 ArtifactStore 工具态记忆；L3 用户画像与跨会话偏好 |
| 上下文 | `ClientContext.prepare_messages_for_invoke()` 动态选择 full/compressed/profile-only |
| Web | FastAPI `POST /api/chat/stream`，SSE 传输 token、工具、A2UI、地图、天气、错误和完成事件 |
| 会话 | `ChatSessionStore` 按 session_id 保存 Agent/context/messages，使用异步锁与 2 小时 TTL |
| 容错 | 120 秒超时、1 次重试、递归上限 20、断连取消、失败时回滚未完成的用户消息 |
| 可视化 | A2UI 表单与地点卡片；高德地图 POI、路线和天气数据 |
| 部署 | Docker Compose 单 worker；同进程 lifespan 启动 FastAPI 与内部 MCP Server |
| 测试 | pytest 覆盖分层编排、记忆、工具筛选、A2UI、SSE、取消与会话连续性 |

## 专题目录

- [01 架构选型](01-architecture-design.md)（36）
- [02 工具管理](02-tool-management.md)（30）
- [03 容错与鲁棒性](03-fault-tolerance.md)（27）
- [04 记忆与上下文](04-memory-context.md)（52）
- [05 评估与全局观](05-eval-and-vision.md)（30）
- [06 多智能体协作](06-multi-agent-collab.md)（22）
- [07 工程化踩坑](07-engineering-pitfalls.md)（49）
- [08 Prompt 工程与框架原理](08-prompt-engineering.md)（18）
- [09 RAG 与检索系统](09-rag-retrieval.md)（66）
- [10 训练、数据与模型优化](10-training-and-data.md)（52）
- [11 AI 代码分析与测试](11-ai-code-testing.md)（8）
- [12 业务 AI 工程分析](12-business-ai-engineering.md)（7）
- [13 简历项目拷打](13-project-deep-dive.md)（23）
- [14 各公司面试偏好](14-company-preferences.md)（索引说明）
- [15 概念考察](15-agent-concepts.md)（13）

## 面试表达原则

- 不说“看情况”，先给判断维度，再映射到 TravelAgent。
- 不把“计划中的能力”说成“已经上线的能力”。
- 先讲请求主链路，再讲失败链路、状态、观测和边界。
- 对未实现能力，使用：“当前版本没有做 X；现有 Y 可以作为基础；生产化会按 Z 落地。”
