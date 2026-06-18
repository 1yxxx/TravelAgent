# TravelAgent — Agent 工程学习指南

> 目标读者：正在系统学习 Agent 开发的后端工程师。
> 阅读方式：理解架构全局 → 跟随请求链路 → 深入各层模式。
> 基础：2026-06-19 重构后的新目录结构。

---

## 目录

1. [项目概览与学习目标](#1-项目概览与学习目标)
2. [架构总览](#2-架构总览)
3. [Agent 生命周期：从请求到响应](#3-agent-生命周期从请求到响应)
4. [Agent 装配：`build_agent()` 工厂模式](#4-agent-装配build_agent-工厂模式)
5. [模型适配层：DeepSeekChatOpenAI](#5-模型适配层deepseekchatopenai)
6. [ReAct Agent 机制](#6-react-agent-机制)
7. [工具工程：分层工具设计](#7-工具工程分层工具设计)
8. [MCP 协议层：工具发现与调用](#8-mcp-协议层工具发现与调用)
9. [上下文工程与三层记忆](#9-上下文工程与三层记忆)
10. [分层编排（可选特性）](#10-分层编排可选特性)
11. [前端数据协议与输出投射](#11-前端数据协议与输出投射)
12. [设计模式总结](#12-设计模式总结)

---

## 1. 项目概览与学习目标

**TravelAgent** 是一个基于 **LangGraph ReAct** 架构的多工具 AI 旅行规划 Agent。它集成高德地图 API，支持自然语言对话式行程规划、POI 搜索、路线规划、天气查询和预算估算，通过浏览器地图实时交互。

### 学完本文后，你将能回答：

1. Agent 在哪里创建？装配了哪些组件？
2. 模型每轮收到了什么消息？context 如何动态变化？
3. 模型如何获知有哪些 Tool？Tool Call 如何路由到 Python 函数？
4. Tool 结果如何重新进入模型上下文？如何持久化？
5. Prompt、记忆和编排分别控制什么？
6. 为什么一次请求会触发多次工具调用？ReAct 循环如何终止？
7. 对话历史、Artifact 和用户画像如何组成三层记忆？
8. 如何将 Agent 输出投射为前端可渲染的数据？

---

## 2. 架构总览

### 2.1 双服务架构

```
┌──────────────────────────────────────────────┐
│  浏览器 (index.html + app.js)                 │
│  ┌────────────────┐  ┌──────────────────────┐ │
│  │ 对话面板        │  │  高德 JS API 地图     │ │
│  └───────┬────────┘  └──────────┬───────────┘ │
└──────────│─────────────────────│──────────────┘
           │ WebSocket            │ JS SDK
┌──────────▼──────────────────────────────────────┐
│  FastAPI (port 8000) ── src/travel_agent/api/   │
│  ├── server.py           应用、lifespan、路由     │
│  ├── websocket_handler.py  WebSocket 消息处理     │
│  ├── a2ui_bridge.py       A2UI 卡片生成与发送     │
│  ├── map_extractor.py     地图/天气 JSON 提取     │
│  ├── message_utils.py     消息序列化与清理        │
│  ├── html_render.py       PDF 导出               │
│  └── file_routes.py       上传/导出路由           │
└──────────┬──────────────────────────────────────┘
           │ langchain-mcp-adapters (HTTP)
┌──────────▼──────────────────────────────────────┐
│  MCP Server (port 8002) ── src/travel_agent/mcp/│
│  ├── server.py            FastMCP 创建与生命周期  │
│  ├── register_tools.py    17 个工具注册           │
│  ├── adapters.py          工具导入映射与工厂       │
│  └── hooks/               拦截器 (耗时/鉴权)      │
└──────────┬──────────────────────────────────────┘
           │ REST API
┌──────────▼──────────────────────────────────────┐
│  高德地图 REST API / DeepSeek LLM API           │
└─────────────────────────────────────────────────┘
```

两个进程通过各自的 `lifespan` 管理生命周期。FastAPI 的 `lifespan` 创建 `asyncio.Task` 启动 MCP Server；关闭时取消 Task。MCP Server 不可用时 Agent 无法工作。

### 2.2 依赖拓扑

```
api/server.py
  └→ api/websocket_handler.py
       └→ agent/factory.py (build_agent)
            ├→ agent/deepseek_adapter.py (DeepSeekChatOpenAI)
            ├→ langchain_mcp_adapters (连接 MCP Server 获取工具)
            ├→ agent/context.py (ClientContext)
            │    ├→ agent/precheck.py
            │    ├→ agent/memory_switch.py
            │    └→ storage/ (L1/L2/L3)
            ├→ agent/node_manager.py (工具元数据)
            │    └→ orchestration/scenario_tags.py
            ├→ orchestration/ (分层编排，可选)
            ├→ skills/ (Markdown Skills)
            └→ mcp/ (工具注册)
                 └→ tools/ (15 个核心工具)
```

**单向依赖**，无循环：`api → agent → orchestration → tools` / `mcp → tools`。

### 2.3 包职责速览

| 包 | 文件数 | 职责 |
|---|---|---|
| `api/` | 9 | Web & CLI 展示层：FastAPI 路由、WebSocket、A2UI、地图提取 |
| `agent/` | 7 | Agent 装配层：工厂、上下文、模型适配、前置校验、NodeManager |
| `tools/` | 19 | 核心工具：search (4) / planning (6) / rendering (2) / utility (2) |
| `mcp/` | 6 | MCP 服务层：Server 创建、工具注册、适配器工厂、拦截器 |
| `orchestration/` | 6 | 分层编排：策略、校验、度量、编排器、场景标签 |
| `storage/` | 5 | 三层记忆：L1 压缩 / L2 ArtifactStore / L3 用户画像 |
| `skills/` | 1 | Skills 热插拔 (skillkit) |
| `utils/` | 3 | Prompt 模板引擎、日志 |

---

## 3. Agent 生命周期：从请求到响应

一次 WebSocket 请求的完整链路（`api/websocket_handler.py`）：

```
用户输入 "帮我规划成都3天行程"
  │
  ▼ WebSocket 接收 → HumanMessage
  │
  ▼ context.precheck_user_request()
  │   校验：意图？目的地？天数？→ 不合格则弹出 form_card
  │
  ▼ context.prepare_messages_for_invoke(messages)
  │   ├── choose_memory_framework()  选择模式 (full/compressed/profile_only)
  │   ├── L3: user_profile 提取偏好
  │   ├── L1: memory_compressor 压缩早期消息（如需要）
  │   └── build_dynamic_system_prompt()  注入 L2/L3
  │
  ▼ agent.ainvoke({"messages": invoke_messages})
  │   └── LangGraph ReAct 循环 (最多 20 步)
  │        ├── LLM 推理 → 决定调用 search_poi
  │        ├── MCP Client → HTTP → MCP Server → search_poi → 高德API
  │        ├── ArtifactStore.save_result() 持久化结果
  │        ├── ToolMessage 写入消息历史
  │        ├── LLM 推理 → 决定调用 check_weather
  │        ├── LLM 推理 → 决定调用 smart_plan_itinerary
  │        ├── LLM 推理 → 决定调用 render_itinerary
  │        └── LLM 推理 → 输出最终文本回复
  │
  ▼ 后端加工 (api/map_extractor.py + api/a2ui_bridge.py)
  │   ├── extract_map_blocks()    提取地图 JSON (itinerary/pois/route)
  │   ├── extract_weather_block() 提取天气数据
  │   └── extract_place_cards()   生成地点卡片
  │
  ▼ clean_messages_for_next_turn()  清理消息历史 (DeepSeek 兼容)
  │
  ▼ WebSocket 推送: 文本回复 + JSON 块 + A2UI 卡片
  │
  ▼ 前端解析渲染: 地图标记 + 行程卡片 + 天气组件
```

**关键设计决策**：Agent 最多重试 1 次（超时 120s），ReAct 循环上限 20 步。LLM 自主决定调用哪些工具——不使用硬编码 `if-else` 调度。

---

## 4. Agent 装配：`build_agent()` 工厂模式

位于 `agent/factory.py`，是项目中最重要的单一函数。

### 4.1 装配步骤

```python
async def build_agent(cfg, session_id, *, lang="zh"):
    # 1. 创建 LLM
    llm = _build_llm(cfg)  # DeepSeek 或标准 ChatOpenAI

    # 2. 通过 MCP Client 获取工具（不直接 import core_nodes）
    client = MultiServerMCPClient(connections={...})
    tools = await client.get_tools()

    # 3. 加载 Markdown Skills
    skills_tools = await load_skills(skill_dir=".storyline/skills")
    tools = tools + skills_tools

    # 4. 创建 ReAct Agent
    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)

    # 5. 可选：包装分层编排
    if cfg.orchestration.enabled:
        agent = LayeredTravelAgent(llm=llm, tools_by_layer=..., ...)

    # 6. 初始化三层记忆
    compressor = MemoryCompressor(llm=llm, session_dir=...)
    user_profile = UserProfileStore(...)
    artifact_store = ArtifactStore(...)

    # 7. 创建运行时上下文
    context = ClientContext(
        cfg=cfg, session_id=session_id,
        memory_compressor=compressor,
        user_profile=user_profile,
        artifact_store=artifact_store,
        ...
    )
    return agent, context
```

### 4.2 设计的工程含义

| 设计 | 类比 Java |
|---|---|
| `build_agent()` | `@Configuration` + `@Bean` 方法（Spring 的 ApplicationContext 装配） |
| `ClientContext` | `@RequestScope` Bean —— 每个会话独立，不跨连接共享 |
| 通过 MCP Client 获取工具 | 服务发现：Agent 不直接依赖工具实现，通过 HTTP 发现 |
| Skills 热插拔 | 插件系统：新增 `SKILL.md` 重启后自动生效 |

---

## 5. 模型适配层：DeepSeekChatOpenAI

位于 `agent/deepseek_adapter.py`。

**问题**：DeepSeek API 不接受 `content: [...]`（list 类型），要求 `content: "string"`。标准 LangChain 在某些消息中会生成 list 类型的 content。

**解决方案**：`DeepSeekChatOpenAI(ChatOpenAI)` 子类覆盖 `_get_request_payload()`：

```python
class DeepSeekChatOpenAI(ChatOpenAI):
    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # 1. 将所有 content 拍平为 string
        # 2. 回注 reasoning_content (DeepSeek thinking 模式)
        for msg in payload["messages"]:
            if not isinstance(msg["content"], str):
                msg["content"] = _flatten_content(msg["content"])
        return payload
```

**两层保障**：
- 发送前：`DeepSeekChatOpenAI._get_request_payload()` 拍平 content
- 接收后：`message_utils.clean_messages_for_next_turn()` 序列化历史消息

---

## 6. ReAct Agent 机制

### 6.1 什么是 ReAct

ReAct (Reasoning + Acting) 是一种 Agent 范式：LLM 在思考后自主决定调用工具，观察工具结果后再决定下一步：

```
System: "你是一个旅行助手，可用工具: search_poi, check_weather, ..."
User: "帮我规划成都3天行程"

→ LLM 思考: 需要搜索成都景点
→ ToolCall: search_poi(city="成都", keyword="景点")
→ ToolResult: [{name: "宽窄巷子", ...}, ...]

→ LLM 思考: 已有景点，还需要天气
→ ToolCall: check_weather(city="成都", forecast=True)
→ ToolResult: {days: [{date: "...", weather: "晴"}, ...]}

→ LLM 思考: 信息足够，规划行程
→ ToolCall: smart_plan_itinerary(spots=..., hotels=..., days=3, ...)
→ ToolResult: {days: [{label: "第1天", spots: [...]}, ...]}

→ LLM 思考: 渲染到地图
→ ToolCall: render_itinerary(days=..., city="成都")

→ LLM 输出: "为您规划了成都3日行程：第一天..."
```

### 6.2 LangGraph 实现

项目使用 `langgraph.prebuilt.create_react_agent()`，它内部维护一个消息列表循环：

1. 将当前消息列表发给 LLM
2. 如果 LLM 返回 `tool_calls`，调用工具并追加 ToolMessage
3. 重复直到 LLM 返回纯文本（无 tool_calls）
4. `recursion_limit=20` 防止死循环

### 6.3 为什么可能重复调用工具

这不是 Bug，是 ReAct 的特性。LLM 可能在收到工具结果后发现信息不足，再次调用同一工具。去重策略在消费端（`map_extractor.py` 的 `_seen_itinerary_keys` / `_seen_pois_keys`），而非在 Agent 执行时拦截。

---

## 7. 工具工程：分层工具设计

### 7.1 工具分类

所有 16 个核心工具按领域分为 4 个子包：

```
tools/
├── search/        外部 API 调用
│   ├── search_poi.py         高德关键字搜索
│   ├── search_hotel.py       酒店搜索 (支持 budget_level)
│   ├── search_restaurant.py  餐厅搜索 (支持菜系)
│   └── check_weather.py      高德天气查询
│
├── planning/      算法 + LLM 加工
│   ├── plan_itinerary.py         简单均分版
│   ├── smart_plan_itinerary.py   K-means 聚类 + 贪心路径 (核心)
│   ├── plan_route.py            驾车路线规划
│   ├── estimate_budget.py       预算估算
│   ├── recommend_transport.py   交通方式建议
│   └── format_itinerary.py      LLM 生成 Markdown 报告
│
├── rendering/     纯数据格式转换 (不调 API)
│   ├── render_map.py   POI 标记 + 路线折线 → 前端 JSON
│   └── render_itinerary.py  行程 → 有序标记 JSON
│
└── utility/       横切工具
    ├── json_tools.py          JSON 校验 + LLM 修复
    └── request_travel_info.py 信息补充表单触发
```

### 7.2 工具定义范式

所有工具使用 LangChain `@tool` 装饰器：

```python
from langchain_core.tools import tool

@tool("search_poi", return_direct=False)
async def search_poi_tool(
    keyword: str,
    city: Optional[str] = None,
    category: Optional[str] = None,
    page_size: int = 10,
) -> List[Dict[str, Any]]:
    """搜索指定城市内的旅游相关 POI。"""  # ← 这个 docstring 就是 LLM 看到的 Tool Description
    cfg = load_settings(default_config_path())
    pois = await _amap_place_text_search(keyword=keyword, api_key=cfg.map.api_key, ...)
    return [{"name": p.get("name"), "longitude": ..., "latitude": ..., ...} for p in pois]
```

**关键点**：`@tool` 的 docstring 直接作为 LLM 的 Tool Description。描述的质量直接影响 LLM 的工具选择准确率。

### 7.3 渲染工具 vs 搜索工具

**搜索工具**调用外部 API，属于领域逻辑层。
**渲染工具**只做数据格式转换（`dict → JSON string`），属于展示层。

分离原因：LLM 不应被"数据打包"细节干扰思考，搜索和规划完成后才渲染。这也使得在前端离线测试时可以直接 mock 渲染输出。

### 7.4 `smart_plan_itinerary` — 核心算法

```
输入: spots[], hotels[], restaurants[], days, city, pace, weather_summary
处理:
  1. K-means 风格迭代聚类 (Haversine 球面距离，最多 10 轮)
  2. 按 pace 控制每天景点数 (relaxed=2/standard=3/intensive=4)
  3. 雨天识别 (正则提取 + 室内优先)
  4. 贪心最近邻排序减少迂回
  5. 跨天去重 + 循环分配酒店和餐厅
输出: {city, title, days: [{label, spots: [{name, lng, lat, note}], hotel, meals}]}
```

---

## 8. MCP 协议层：工具发现与调用

### 8.1 为什么需要 MCP

MCP (Model Context Protocol) 在本项目中扮演双重角色：
1. **内部工具调用的统一通道**：Agent 不直接 `import` 工具函数，而是通过 HTTP 调用
2. **对外暴露工具**：外部 MCP 客户端（如 Claude Desktop）可直接连接使用

### 8.2 调用链路

```
LLM ToolCall "search_poi"
  → LangGraph MCP Tool (langchain-mcp-adapters)
  → HTTP POST http://127.0.0.1:8002/mcp
      Headers: {"X-Travel-Session-Id": "travel_xxx"}
  → FastMCP Server
  → register_tools.py: mcp_search_poi()
  → _execute_tool("search_poi", ctx, cfg, invoke_args)
       ├── adapters._import_tool("search_poi") → 延迟导入 core tool
       ├── core_tool.ainvoke(args) → 高德 API
       └── store.save_result() → 持久化到 ArtifactStore
  → 返回 MCP 信封: {artifact_id, result, isError}
```

### 8.3 工具注册（工厂模式消除模板）

重构前，`register_tools.py` 有 15 个几乎完全相同的 `@server.tool()` 块（~25KB）。重构后：

```python
# 每个工具只需声明参数签名 + 委托到共享执行体
@server.tool(name="search_poi", description="...")
async def mcp_search_poi(ctx, city, keyword, types=None, max_results=5):
    return await _execute_tool("search_poi", ctx, cfg, {"city": city, ...})
```

共享执行体 `_execute_tool()` 统一处理：**延迟导入 → 调用 → save_result → 返回信封**。

### 8.4 会话隔离

每个工具调用携带 `X-Travel-Session-Id` 请求头，`adapters._get_store(ctx, cfg)` 通过 `SessionLifecycleManager` 获取该会话的 `ArtifactStore`，确保不同用户的搜索、天气和行程结果互不污染。

---

## 9. 上下文工程与三层记忆

### 9.1 什么是上下文工程

Agent 的"上下文" = System Prompt + 消息历史 + 记忆注入。`ClientContext.prepare_messages_for_invoke()` 是上下文的统一装配入口。

### 9.2 动态上下文构建流程

```python
async def prepare_messages_for_invoke(self, messages):
    # 1. 估算 token → 选择记忆模式
    mode, reason = choose_memory_framework(
        message_count=len(messages),
        token_estimate=...,
        soft_threshold=24/3500,   # 达到即触发 L1 压缩
        hard_threshold=60/7000,   # 达到即关闭 L2，仅保留 L3
    )

    # 2. L3: 用户偏好提取 (轻量规则)
    self.user_profile.extract_preferences_from_messages(messages)

    # 3. L1: 消息压缩 (如果触发阈值)
    if mode in {"compressed_context", "profile_only"}:
        messages = await self.memory_compressor.maybe_compress(messages)

    # 4. 动态拼装 system prompt
    dynamic_prompt = self.build_dynamic_system_prompt(store=...)
    return [SystemMessage(content=dynamic_prompt)] + messages
```

### 9.3 三层记忆架构

| 层级 | 组件 | 文件 | 生命周期 | 注入方式 |
|---|---|---|---|---|
| **L1** | `MemoryCompressor` | `storage/memory_compressor.py` | 单 session | LLM 摘要压缩早期消息 → `SystemMessage` |
| **L2** | `ArtifactStore` | `storage/agent_memory.py` | 单 session | `build_context_prompt()` 快照注入 system prompt |
| **L3** | `UserProfileStore` | `storage/user_profile.py` | 跨 session | 规则提取偏好 → system prompt 前缀 |

### 9.4 动态 System Prompt 结构

```
┌──────────────────────────────────────────┐
│ 原始 system prompt (指令 + 工具说明)       │  ← prompts/tasks/instruction/zh/system.md
├──────────────────────────────────────────┤
│ L3: 用户偏好    "用户偏好城市：成都、杭州"   │  ← UserProfileStore.build_profile_prompt()
│                  "预算：mid"               │
│                  "节奏：standard"          │
├──────────────────────────────────────────┤
│ L2: 工具快照    "已搜索 POI (5个):          │  ← ArtifactStore.build_context_prompt()
│                   宽窄巷子, 锦里, ..."      │
│                  "已查天气: 晴 22°C"       │
└──────────────────────────────────────────┘
```

### 9.5 memory_switch 策略

```toml
[memory_switch]
enabled = true
soft_message_threshold = 24    # 24 条消息 → 触发 L1 压缩
hard_message_threshold = 60    # 60 条消息 → 关闭 L2，仅保留 L3
soft_token_threshold = 3500
hard_token_threshold = 7000
```

三档模式：`full_context` → `compressed_context` → `profile_only`。目的是在上下文窗口限制下保住最关键信息。

---

## 10. 分层编排（可选特性）

位于 `orchestration/`，默认关闭（`config.toml` 中 `[orchestration].enabled = false`）。

### 10.1 五层流水线

```
requirement → research → planning → risk → render
  需求确认      搜索/天气    规划/编排    风险校验   渲染输出
```

每层只暴露该层的工具白名单（由 `NodeManager` 管理）。`LayerValidator` 做最小校验（research 层必须触发搜索工具，planning 层必须触发编排工具）。失败时回滚到最近 checkpoint 重试。

### 10.2 与自由 ReAct 的对比

| 维度 | 自由 ReAct | 分层编排 |
|---|---|---|
| 工具选择 | LLM 自由决定 | 每层白名单限制 |
| 执行保证 | 依赖 prompt 提示 | LayerValidator 强制校验 |
| 失败处理 | 继续或卡死 | 回滚 + 重试 |
| 成本 | 低延迟 | 每层额外 LLM 调用 |

当前默认关闭是因为回滚时状态等价性导致偶发死锁，适合作为学习分层 Agent 架构的参考实现。

### 10.3 场景标签（消除代码重复）

`orchestration/scenario_tags.py` 提供 `infer_scenario_tags()` 函数，从用户文本中识别 11 种场景标签（family/senior/couple/solo/budget/luxury/outdoor/culture/short_trip/long_trip/custom）。该函数同时被 `LayeredTravelAgent` 和 `NodeManager` 使用——这就是为什么它被提取到 `orchestration/` 包中而不是放在某个调用方内。

---

## 11. 前端数据协议与输出投射

### 11.1 JSON 块协议

后端将地图/天气数据以 markdown 代码块形式附加到 LLM 回复末尾：

````markdown
助手: 为您规划了成都3日行程...

```json
{"__type": "itinerary", "city": "成都", "title": "...", "days": [...]}
```

```json
{"__type": "weather", "city": "成都", "days": [{...}]}
```
````

前端 `web/static/app.js` 通过正则提取 `__type` 字段进行渲染：

| `__type` | 来源工具 | 前端渲染 |
|---|---|---|
| `pois` | `render_map_pois` / search 兜底 | 标记点 |
| `itinerary` | `render_itinerary` / `smart_plan_itinerary` | 分组标记 + 连线 + 卡片 |
| `route` | `render_map_route` / `plan_route` | 折线 |
| `weather` | `check_weather` | 天气组件 |

### 11.2 兜底机制

`map_extractor.py` 中的 `extract_map_blocks()` 包含一个关键兜底逻辑：当 LLM 做了搜索（`search_poi`）但忘了调用 `smart_plan_itinerary` 或 `render_itinerary` 时，后端会自动用搜索结果构建一个 `itinerary` 块。这解决了 LLM 工具调用不完整的常见问题。

### 11.3 输出加工流水线

```
Agent 原始输出
  │
  ▼ api/message_utils.normalize_content()   序列化 content
  ▼ api/map_extractor.extract_map_blocks()   提取地图 JSON
  ▼ api/map_extractor.extract_weather_block() 提取天气 JSON
  ▼ api/a2ui_bridge.extract_place_cards()    生成地点卡片
  ▼ api/message_utils.clean_messages_for_next_turn()  清理历史
  │
  ▼ WebSocket 推送 → 前端渲染
```

---

## 12. 设计模式总结

| 模式 | 体现位置 | 作用 |
|---|---|---|
| **Factory** | `agent/factory.py` — `build_agent()` | 一站式装配 LLM + Tools + Memory + Context |
| **Request Scope** | `agent/context.py` — `ClientContext` | 每会话独立的运行时上下文 |
| **Facade** | `mcp/` — FastMCP Server | 统一工具调用入口 |
| **Adapter** | `agent/deepseek_adapter.py` / `mcp/adapters.py` | 协议适配 (DeepSeek / MCP) |
| **Strategy** | `agent/memory_switch.py` — `choose_memory_framework()` | 三套记忆模式动态切换 |
| **Repository** | `storage/agent_memory.py` — `ArtifactStore` | 工具结果持久化 |
| **Template Method** | `utils/prompts.py` — `PromptBuilder` | Markdown 模板 + `{{variable}}` 替换 |
| **Plugin** | `skills/skills_io.py` | Skills 热插拔 |
| **Pipeline** | `orchestration/layered_agent.py` | 五层流水线编排 |

### 工程实践建议

1. **工具签名即契约**：Tool 的 docstring 是 LLM 的 API 文档，改签名要同步更新 description
2. **延迟导入**：MCP 工具通过 `_import_tool()` 延迟导入，避免模块级循环依赖
3. **去重在消费端**：不在 Agent 执行时拦截重复调用，而在 `map_extractor` 中按 key 去重
4. **DeepSeek 兼容**：两处 content 拍平（发送前 + 历史清理），新增模型时先检查能否接受 list content
5. **config.toml 是唯一配置源**：Pydantic `extra="forbid"` 防止拼写错误，相对路径以配置文件目录为基准
