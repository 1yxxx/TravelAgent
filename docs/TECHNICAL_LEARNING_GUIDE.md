# TravelAgent — Agent 面试题精解

> 面试题来源：[zero2Agent 面试通关](https://onefly.top/zero2Agent/learn-agent-interview)
> 回答方式：结合 TravelAgent 项目**实际做了什么**，而非代码片段

---

## 一、LangGraph ReAct 状态流转

### Q1 你用 ReAct 还是 Plan-and-Execute？为什么？

**回答**：本项目**默认用 ReAct**，同时实现了一个**实验性的 Plan-and-Execute 变体**（分层编排），但默认关闭。

**实际实现**：

项目使用 `langgraph.prebuilt.create_react_agent()` 创建标准 ReAct Agent。Agent 内部维护一个消息列表作为共享状态，LLM 每次推理后如果决定调用工具，工具执行结果自动追加到消息列表，LLM 再次推理，直到输出纯文本结束。

选择 ReAct 的原因是旅行规划场景中用户需求多变——有人先问酒店再补景点，有人直接说"3 天成都"。LLM 自主决定调用什么工具、以什么顺序调用，比固定流水线更灵活。

**Plan-and-Execute 变体**（分层编排）的实现：把一次规划拆成 5 层（需求确认→搜索调研→行程规划→风险校验→渲染输出），每层只暴露该层允许的工具。比如 research 层只有搜索和天气工具，planning 层只有规划工具。每层执行后做最小校验——research 层必须触发过搜索类工具，planning 层必须触发过编排类工具。失败则回滚到上一个 checkpoint 重试。

这个变体默认关闭的原因：回滚后 LLM 可能做出相同决策再次失败，形成死锁。

---

### Q2 LangGraph 中的 State 怎么定义和流转？节点多了怎么防止状态膨胀？

**回答**：

**状态定义**：项目使用 `create_react_agent` 预构建 Agent，它内部使用 `MessagesState`——一个带 `add_messages` reducer 的消息列表。Reducer 的作用是：多个节点写入同一字段时，消息是**追加**而非覆盖。

**状态流转**：Agent 图有 3 个核心节点：
- `agent` 节点：调用 LLM，输出 AIMessage（可能含 tool_calls）
- `tools` 节点：执行工具，输出 ToolMessage
- 条件边：如果 LLM 返回 tool_calls → 路由到 tools；如果 LLM 返回纯文本 → 路由到 END

每次状态变化是追加式的——AIMessage 追加到列表末尾，ToolMessage 再追加，LLM 始终能看到完整历史。

**防止状态膨胀的 4 层机制**：

第一层：`recursion_limit=20`。单次 Agent 推理最多走 20 轮 Agent→Tools 循环，超过 LangGraph 直接抛异常。

第二层：L1 消息压缩。当消息数超过 24 条时，把早期消息（保留最近 10 条）交给 LLM 生成一段摘要，用摘要 SystemMessage 替换早期消息。这样消息列表长度不会无限增长。

第三层：动态 System Prompt。每轮对话前重建 SystemMessage，不把它写回历史。历史中只保留 HumanMessage、AIMessage、ToolMessage，SystemMessage 每轮都是新的。

第四层：硬阈值保护。当消息数超过 60 条或 token 估算超过 7000 时，关闭 L2（ArtifactStore 快照）注入，只保留 L3 用户偏好，进一步节省 token。

---

### Q3 生产级 Agent 的执行循环包含哪些阶段？哪些必须显式状态化？

**回答**：本项目的一次请求经历 7 个阶段，其中 4 个必须持久化状态：

**阶段 1：前置校验**
- 做的事：检查用户输入是否有旅行意图、是否包含目的地、是否指定天数
- 状态化：不需要——纯判断函数，无状态

**阶段 2：上下文构建**
- 做的事：估算 token → 选记忆模式 → 可能压缩历史 → 提取用户偏好 → 拼装动态 System Prompt
- 状态化：需要。messages 列表是累积的对话历史；L1 摘要写入 `summary.json`；L3 偏好写入 `user_profiles/default.json`

**阶段 3：ReAct 循环**
- 做的事：LLM 反复推理和调用工具，直到输出纯文本
- 状态化：需要。LangGraph 内部维护 messages 状态列表，每次工具调用结果追加进去

**阶段 4：工具结果持久化**
- 做的事：每次工具调用后，MCP 层自动把结果写入 `travel_outputs/<session_id>/` 目录树
- 状态化：需要。每个工具调用结果存为独立 JSON 文件，`meta.json` 维护索引

**阶段 5：结果提取**
- 做的事：从 ToolMessage 中扫描地图数据、天气数据、地点卡片
- 状态化：不需要——纯数据转换

**阶段 6：历史清理**
- 做的事：把消息列表中的 list/dict 类型 content 序列化为 string（DeepSeek 兼容）
- 状态化：不需要——纯转换

**阶段 7：响应推送**
- 做的事：通过 WebSocket 发送回复文本 + 地图 JSON 块 + A2UI 卡片事件
- 状态化：不需要——传输层

---

### Q4 AgentState 的作用是什么？为什么不使用全局变量？

**回答**：LangGraph 的 AgentState 是贯穿整个图执行的共享数据容器。

**本项目为什么不使用全局变量**：每个 WebSocket 连接有独立的 `messages = []` 列表。如果用全局变量，不同用户的数据会互相污染。AgentState 提供了额外的工程保障：

1. **Reducer 语义**：`add_messages` reducer 确保多个节点写入同一字段时是追加而非覆盖——这很重要，因为 Agent 执行中 agent 节点和 tools 节点都要往 messages 里写东西
2. **可回溯**：分层编排中利用 checkpoint 快照实现回滚——每层成功后 `list(messages)` 保存快照，失败时恢复到上一个成功点
3. **可序列化**：状态可以被持久化到磁盘，支持跨进程恢复。项目虽未使用 LangGraph 的 SqliteSaver，但通过 ArtifactStore 实现了等效的工具结果持久化

---

### Q5 ReAct 循环中如何纠正逻辑塌缩或无效工具调用？

**回答**：三层纠正策略：

**第一层：System Prompt 引导**。System Prompt 中规定了推荐的 7 步工具调用顺序（先搜索景点→查天气→搜酒店→搜餐厅→智能规划→渲染→格式化），但不强制执行。

**第二层：recursion_limit 硬性约束**。最多 20 轮循环，超过就抛异常，防止 LLM 无限循环。

**第三层：后端兜底纠正（最关键）**。当 LLM 搜了景点但忘了调用 `smart_plan_itinerary` 或 `render_itinerary` 时，后端的地图提取模块会自动检测到"有搜索结果但没有行程块"，然后**后端直接调用** `smart_plan_itinerary` 算法，用搜索结果生成结构化行程并注入到回复中。用户完全无感知——即使 LLM "偷懒"，地图上也会展示完整的行程标注。

分层编排模式还增加了额外的校验：每层执行后用 LayerValidator 检查该层是否调用了应该调用的工具类型，不通过就回滚重试。

---

### Q6 LangGraph 和 LangChain 的区别？分别适合什么场景？

**回答**：本项目同时用了两者，职责分明：

**LangChain 在本项目中的用途**：
- 工具定义（`@tool` 装饰器把普通 Python 函数变成 LLM 可调用的 Tool）
- 消息类型（`HumanMessage`、`AIMessage`、`ToolMessage`、`SystemMessage`）
- LLM 接口封装（`ChatOpenAI` 统一了不同模型提供商的调用方式）

**LangGraph 在本项目中的用途**：
- 提供 ReAct 循环——"LLM 推理→工具调用→追加结果→LLM 再推理"这个循环，LangChain 的链式模型做不到
- 状态管理——`MessagesState` + `add_messages` reducer 维护跨节点的消息列表
- 条件路由——根据 LLM 是否返回 tool_calls 决定下一步走 tools 节点还是 END

**适合场景**：LangChain 适合单次调用（"查一下天气"），LangGraph 适合需要多轮推理和工具调用的复杂任务（"帮我规划成都 3 天行程"）。

---

### Q7 如何限制 Agent 的思考深度、工具调用次数和递归层级？

**回答**：四层限制：

1. `recursion_limit=20`：LangGraph 在达到限制后抛异常，Agent 循环终止
2. `asyncio.wait_for(timeout=120)`：单次请求 120 秒超时，防止挂死
3. `MAX_RETRIES=1`：总共最多 2 次尝试（1 次正常 + 1 次重试）
4. 记忆硬阈值：消息数 ≥ 60 或 token ≥ 7000 时关闭 L2 注入，节省 token 避免上下文爆炸

分层编排中还有每层的重试上限：`max_retries_per_layer=1`，每层最多重试 1 次。

---

### Q8 LangGraph 的 State Snapshot 机制怎么实现？

**回答**：本项目的分层编排中利用了 checkpoint 快照机制：

- 执行开始前保存 `checkpoints["start"] = list(messages)` 作为初始快照
- 每层校验通过后保存 `checkpoints[layer] = list(messages)`
- 校验失败时恢复到上一个成功的 checkpoint：`messages = list(checkpoints[last_ok_key])`
- 恢复后注入一条 HumanMessage："阶段 X 校验失败，请重试"，引导 LLM 改变决策路径

普通 ReAct 模式不使用 LangGraph 的 SqliteSaver 持久化 checkpoint，而是通过 WebSocket 层的 `messages` 列表 + ArtifactStore 的工具结果持久化实现等效的跨轮次状态保持。

---

### Q9 你的项目有没有用到 ReAct 模式？怎么用的？

**回答**：**是的，ReAct 是项目的唯一执行模式**。

一次真实的"规划成都 3 天行程"的 ReAct 循环：

1. LLM 收到用户请求 → 思考"需要搜索成都景点" → 调用 `search_poi`
2. `search_poi` 返回 5 个 POI → 结果追加到消息历史
3. LLM 看到结果 → 思考"还需要天气" → 调用 `check_weather`
4. `check_weather` 返回 3 天预报 → 结果追加
5. LLM 思考"信息够了，开始规划" → 调用 `smart_plan_itinerary`
6. `smart_plan_itinerary` 返回结构化行程 → 结果追加
7. LLM 思考"渲染到地图" → 调用 `render_itinerary`
8. LLM 输出纯文本回复 → 循环结束

整个过程 LLM 自主决定调用什么工具、何时调用、以什么顺序调用。System Prompt 只提供建议顺序，不强制执行。

---

## 二、工具管理与 MCP

### Q10 MCP 协议的完整调用过程是怎样的？

**回答**：4 个阶段：

**阶段 1：服务启动**。FastAPI 启动时，`lifespan` 在后台启动 MCP Server（Uvicorn 监听 8002 端口）。MCP Server 创建时，一次性注册全部 17 个工具，同时初始化 SessionLifecycleManager（负责按 session_id 管理 ArtifactStore）。

**阶段 2：工具发现**。每个 WebSocket 连接建立时，`build_agent()` 通过 `MultiServerMCPClient` 向 `http://127.0.0.1:8002/mcp` 发起 `tools/list` 请求，获取所有已注册工具的 JSON Schema。返回的 Tool 对象被合并到 Agent 的工具列表中。

**阶段 3：工具调用**。LLM 决定调用工具时，LangGraph 通过 langchain-mcp-adapters 发起 HTTP POST 到 `/mcp`，Body 中包含 `{"method": "tools/call", "params": {"name": "search_poi", "arguments": {...}}}`。请求头携带 `X-Travel-Session-Id`，MCP Server 据此找到当前会话的 ArtifactStore。

**阶段 4：结果回传**。MCP Server 执行工具后，返回统一信封：`{"artifact_id": "art_abc", "result": [...], "isError": false}`。LangGraph 将其包装为 ToolMessage 追加到消息历史。

---

### Q11 MCP Server 是怎么构建的？

**回答**：3 步构建：

1. **定义 session 生命周期**：通过 `@asynccontextmanager` 创建 lifespan，在 lifespan 中初始化 `SessionLifecycleManager`（管理 `session_id → ArtifactStore` 映射，自动清理过期会话）。lifespan 的返回值会作为 MCP 请求上下文注入到每个工具调用中。

2. **创建 FastMCP 实例**：配置 server name（"travel"）、stateless_http（False——有状态，因为要按 session 隔离数据）、json_response（True）、lifespan。

3. **注册工具**：调用 `register_tools.register()` 遍历 17 个工具，每个工具注册时声明参数签名（含 Annotated + Field description），FastMCP 自动生成 JSON Schema。

**与普通 REST API 的关键区别**：MCP Server 的工具发现是自动的（`tools/list` 端点），Schema 从 Python 函数签名自动生成，调用通过统一端点路由而非每个工具独立 URL。

---

### Q12 MCP 和 Skills 的本质区别是什么？

**回答**：项目同时使用两者，在不同层工作：

**MCP** 是通信协议层——解决"工具如何被发现和调用"的问题。项目通过 MCP Server 暴露 17 个原子工具（`search_poi`、`check_weather` 等），Agent 通过 HTTP 发现和调用它们。

**Skills** 是业务能力层——解决"如何描述多步骤工作流"的问题。项目有 4 个 Markdown 格式的 Skill（`full_trip_planner`、`hotel_recommender` 等），每个 Skill 描述了多步策略。例如 `full_trip_planner` 描述了 5 步流程：确认需求→搜索→规划→渲染→生成报告。

LLM 看到的是：17 个 MCP 工具 + 4 个 Skill 工具。Skill 工具在 LLM 看来也是一个"工具"，但它内部包含了多步策略描述。LLM 可以调用 Skill（让系统自动执行 5 步）而不是手动逐个调用 5 个工具。

**加载方式**：MCP 工具通过 HTTP 从 MCP Server 获取；Skills 从 `.storyline/skills/` 目录的 Markdown 文件加载，通过 `skillkit` 库转换为 LangChain Tool。

---

### Q13 工具描述写得再好，模型也瞎传参数怎么办？

**回答**：三层防线：

1. **MCP JSON Schema 自动校验**：FastMCP 根据函数的 `Annotated[str, Field(...)]` 声明自动生成 JSON Schema，类型不匹配会在 MCP 层被拦截。

2. **默认值兜底**：所有可选参数都有默认值。比如 `max_results=5`、`budget_level="mid"`、`page_size=10`。LLM 不传这些参数也能正常工作。

3. **业务容错**：Core Tool 内部做防御性处理。比如经纬度解析失败时不中断，只是该 POI 没有坐标；高德 API 返回空结果时返回 `[]` 而非报错；MCP 层捕获所有异常返回 `{"isError": true}` 而非崩溃。

---

### Q14 MCP 接入多个工具，返回格式不统一怎么处理？

**回答**：统一信封 + 两层反序列化。

**统一信封**：无论底层工具返回什么（list、dict、JSON string），MCP 层统一包装为 `{"artifact_id": "...", "result": ..., "isError": false}`。

**两层反序列化**：消费端先解包 MCP 信封取出 `result`，再判断 `result` 是不是 JSON string——如果是（render 类工具返回的是 `json.dumps()` 的结果），再做第二次 `json.loads()`。

这个设计确保了搜索工具返回的 POI 列表、规划工具返回的 dict、渲染工具返回的 JSON string 都能被消费端统一处理。

---

### Q15 你的 Agent 有哪些工具？工具是怎么设计的？

**回答**：17 个工具按功能分为 4 类：

**搜索类（4 个）**：`search_poi`、`search_hotel`、`search_restaurant`、`check_weather`。都调用高德 API，返回结构化数据（list/dict）。

**规划类（6 个）**：`plan_itinerary`（简单均分）、`smart_plan_itinerary`（K-means 聚类+贪心排序，核心工具）、`plan_route`（驾车路线）、`estimate_budget`（预算估算）、`recommend_transport`（交通建议）、`format_itinerary`（LLM 生成 Markdown 报告）。

**渲染类（2 个）**：`render_map_pois`、`render_map_route`、`render_itinerary`。不调外部 API，只做数据格式转换——把工具结果打包成前端可解析的 JSON 结构（含 `__type` 字段标记类型）。

**横切类（2 个）**：`validate_json`、`fix_json`、`request_travel_info`（弹表单让用户补充信息）、`read_artifact`（读取之前工具调用的持久化结果）。

**设计原则**：单一职责（每个工具只做一件事）、参数精简（可选参数有默认值）、输出结构化（返回 dict/list 而非自然语言）、渲染工具与业务工具分离（LLM 不关心数据打包细节）。

---

### Q16 工具多导致 token 数过多怎么解决？

**回答**：项目不做动态工具加载（17 个工具对 LLM 全部可见），但通过两种机制控制：

1. **分层编排（可选）**：启用后每层只暴露该层工具。research 层只有搜索+天气工具，planning 层只有规划工具，render 层只有渲染工具。每层 LLM 看到的工具列表大幅减少。

2. **场景标签裁剪**：NodeManager 根据用户文本中的关键词识别场景（家庭/老人/情侣/穷游/豪华/长线等 11 种），然后过滤不相关的工具。比如用户说"穷游"，预算类工具保留但豪华酒店搜索被裁剪。

---

### Q17 多工具场景下怎么保证参数提取准确？

**回答**：三个层面的保障：

1. **MCP 层**：FastMCP 根据 Python 函数签名 + Field description 自动生成 JSON Schema，LLM 收到的 tool definition 包含每个参数的类型和描述，减少传错类型的概率。

2. **默认值设计**：所有可选参数都有合理默认值（`max_results=5`、`budget_level="mid"`），LLM 不传也能正常工作。

3. **Core Tool 容错**：搜索工具内部对高德 API 返回的异常数据做防御性处理（经纬度解析失败不影响其他字段、空结果返回 `[]` 不报错）。

---

## 三、记忆与上下文工程

### Q18 会话记忆具体是怎么实现的？滑动窗口设几轮？摘要压缩怎么触发？

**回答**：双层架构——Buffer + Summary。

**Buffer**：保留最近 10 条消息不压缩，确保 LLM 能看到最近的对话上下文。

**Summary**：当消息总数超过 24 条时触发压缩。把早期消息交给 LLM 生成一段摘要，用摘要 SystemMessage 替换早期消息。摘要持久化到 `session_dir/summary.json`，断线重连后可以恢复。

**触发条件**：不是按轮次，而是按消息数量和 token 估算双阈值判断。软阈值 24 条消息或 3500 token → 触发 L1 压缩；硬阈值 60 条消息或 7000 token → 进一步关闭 L2 注入。

---

### Q19 讲一下 Agent 中的"长短期记忆"

**回答**：项目实现了三层记忆，生命周期不同：

**L1 短期记忆（MemoryCompressor）**：单 session 内有效。对话太长时，把早期消息压缩为 LLM 生成的摘要。摘要存在 `session_dir/summary.json`。

**L2 中期记忆（ArtifactStore）**：单 session 内有效。每次工具调用的完整结果持久化到 `travel_outputs/<session_id>/` 目录树。每个工具调用一个 JSON 文件，`meta.json` 维护索引。通过 `build_context_prompt()` 将最新快照注入 System Prompt，让 LLM 知道"已经搜了什么、查了什么天气"。

**L3 长期记忆（UserProfileStore）**：跨 session 有效。从用户消息中规则提取偏好（城市、预算、节奏），持久化到 `travel_data/user_profiles/default.json`。下次对话时自动注入 System Prompt。

---

### Q20 你怎么理解 Agent 里的"状态"而不是"上下文"？

**回答**：

**上下文**是 LLM 的输入——每轮对话前动态拼装的 System Prompt + 消息历史。变化方式：每轮重建 System Prompt，追加新消息。

**状态**是系统的结构化数据——工具调用结果、用户偏好、对话摘要。变化方式：工具调用后写入 ArtifactStore，偏好提取后写入 UserProfileStore。

关键区别：状态可以**不全部放进上下文**。项目通过 `build_dynamic_system_prompt()` 选择性地把状态快照注入 System Prompt——正常模式下注入 L2 快照 + L3 偏好，硬阈值模式下只注入 L3 偏好，关闭 L2 以节省 token。状态存在于文件系统中，不随上下文变化而丢失。

---

### Q21 长上下文里怎么让 Agent 不忘记关键信息？

**回答**：通过 L2 ArtifactStore 的 System Prompt 注入机制。

每轮对话前，`build_dynamic_system_prompt()` 会读取 ArtifactStore 的 `meta.json` 索引，生成一段"已收集数据"快照注入到 System Prompt 中。例如："已搜索 POI：宽窄巷子、锦里、武侯祠（5个结果）；已查天气：成都 3 天预报（晴 22°C）"。

这意味着即使用户对话了 20 轮、L1 已经把早期消息压缩成摘要，LLM 依然能在 System Prompt 中看到"前期收集了什么数据"，不会因为上下文太长而忘记已经搜过什么。

---

### Q22 上下文窗口不够用怎么办？

**回答**：三档策略：

**正常模式**：全量上下文 + L2 快照 + L3 偏好。

**压缩模式**（消息数 ≥ 24 或 token ≥ 3500）：L1 压缩早期消息为摘要，保留最近 10 条。L2 快照和 L3 偏好继续注入。

**极限模式**（消息数 ≥ 60 或 token ≥ 7000）：L1 压缩 + 关闭 L2 注入（不再注入工具结果快照），只保留 L3 偏好。此时 LLM 看到的上下文最小——压缩后的历史 + 用户偏好，但不再有"已收集数据"快照。

---

### Q23 压缩过程中会丢失工具调用历史，导致模型重复调用工具怎么解决？

**回答**：三个互补方案：

1. **L2 注入不受 L1 压缩影响**。L2 快照是独立于消息历史的——它从文件系统读取，注入到 System Prompt。即使 L1 压缩了消息历史，System Prompt 中仍然有"已收集数据"快照。

2. **L1 压缩只压缩对话文本**，不压缩工具调用结果。压缩 prompt 针对的是 HumanMessage 和 AIMessage 的文本内容。

3. **后端兜底**。如果 LLM 确实忘了调用渲染工具，后端的地图提取模块会检测到"有搜索结果但无行程块"，直接调用 `smart_plan_itinerary` 生成行程并注入到回复中。

---

### Q24 什么是"工具态记忆"（Tool-state Memory）？

**回答**：项目的 L2 ArtifactStore 就是工具态记忆。

存储四类内容：
- **工具调用结果**：`search_poi` 返回的完整 POI 列表
- **调用元数据**：调用时间、tool name、artifact_id
- **调用摘要**：如 "POI 搜索: 景点 @ 成都"
- **引用索引**：`meta.json` 维护所有 artifact 的关系

**作用**：避免 LLM 重复调用工具。System Prompt 中注入了"已搜索 POI: 5个结果"，LLM 知道不需要再搜一次。同时支持 `read_artifact` 工具，LLM 可以按 ID 读取之前工具调用的完整结果。

---

### Q25 如何设计三层记忆机制？

**回答**：项目的三层记忆设计：

**L1 — 滑动窗口 + LLM 摘要**。消息超过 24 条时，早期消息被 LLM 压缩为摘要，最近 10 条保留原文。摘要持久化到 `summary.json`。

**L2 — 工具结果文件系统存储**。每次工具调用后自动写入 `travel_outputs/<session_id>/<tool_name>/<artifact_id>.json`。`meta.json` 维护索引。通过 `build_context_prompt()` 生成快照注入 System Prompt。

**L3 — 跨 session 用户画像**。从消息中规则提取偏好（正则匹配城市名、预算关键词），持久化到 JSON 文件。下次对话自动注入 System Prompt。

**协同工作**：每轮对话前，先选记忆模式（正常/压缩/极限），然后按模式决定是否触发 L1 压缩、是否注入 L2 快照、是否注入 L3 偏好。最终拼装为一个动态 System Prompt 发给 LLM。

---

## 四、容错与鲁棒性

### Q26 如果 Agent 的决策出错了，怎么防范？

**回答**：四层防范：

**前置校验**：检查用户输入是否有旅行意图、是否包含目的地和天数。缺失则弹出表单让用户补充，而非让 LLM 瞎猜。

**执行中约束**：recursion_limit=20 防止死循环，timeout=120s 防止卡死，MAX_RETRIES=1 防止无限重试。

**执行后兜底**：LLM 搜了景点但忘了调规划/渲染工具时，后端直接调用算法生成行程并注入回复。用户无感知。

**分层编排校验**：每层执行后检查该层是否调用了应该调用的工具类型，不通过则回滚重试。

---

### Q27 Agent 如何减少幻觉？

**回答**：旅行场景的幻觉治理：

**生成前**：System Prompt 明确要求"必须先调用工具获取真实数据，不得凭空编造地点信息"。

**生成中**：所有地点信息来自高德 API 返回值（经纬度、名称、地址），不是 LLM 的 parametric knowledge。LLM 只能基于 tool result 做加工和排版，不能"想象"一个不存在的景点。

**生成后**：后端兜底——即使 LLM 没输出完整结果，地图提取模块会自动生成 itinerary 块，确保用户总能看到地图标注。

---

### Q28 上下文爆炸或工具循环调用怎么解决？

**回答**：五个机制协同：

1. `recursion_limit=20`——硬性限制循环步数
2. `asyncio.wait_for(timeout=120)`——超时终止
3. L1 压缩——消息超过 24 条时压缩早期消息
4. 硬阈值——消息超过 60 条时关闭 L2 注入
5. 去重——地图提取时按城市+天数去重，相同行程只保留最新版本

---

### Q29 Agent 系统的 fallback 是怎么做的？

**回答**：四级 fallback：

**工具级**：高德 API 失败 → 返回空列表不抛异常；MCP 层异常捕获 → 返回 `{"isError": true}`。

**编排级**：Skill 加载失败 → 记录 warning 继续运行；分层编排包装失败 → 回退到标准 ReAct。

**请求级**：超时或异常 → 重试 1 次；全部失败 → 发送错误消息给用户。

**输出级**：LLM 忘调渲染工具 → 后端直接调用规划算法生成行程。

---

### Q30 执行到一半工具调超时了 Agent 怎么处理？

**回答**：外层 `asyncio.wait_for(timeout=120)` 超时后抛出 `TimeoutError`，WebSocket 层捕获后向用户发送"请求超时，正在重试"提示，然后用相同的 `invoke_messages` 重试一次。重试不会丢失上下文——`invoke_messages` 在超时前已经构建好，保持不变。

---

### Q31 Agent 用状态机编排时卡死/死循环怎么排查和熔断？

**回答**：分层编排有两层熔断：

**层内熔断**：每层最多重试 `max_retries_per_layer` 次（默认 1 次），超过则停止该层。

**全局熔断**：某层失败后整个管线停止（`break`），不再继续后续层。

再加 WebSocket 层的 120s 超时——即使分层编排内部卡死，外层也会超时终止。

---

### Q32 Agent 做多轮工具调用和单轮相比有哪些额外挑战？

**回答**：5 个挑战及对应解法：

| 挑战 | 解法 |
|---|---|
| 错误累积 | 单轮失败返回 isError 而非中断 |
| 上下文膨胀 | L1 压缩 + 硬阈值 |
| 依赖管理 | System Prompt 建议的 7 步顺序 |
| 状态一致性 | L2 快照注入确保 LLM 知道已收集数据 |
| 延迟叠加 | 异步调用 + 120s 全局超时 |

---

## 五、架构设计与工程实践

### Q33 Agent 的架构设计？从系统角度来拆分

**回答**：按系统职责分为 5 层：

**展示层（api/）**：FastAPI 路由、WebSocket 通信、A2UI 卡片生成、地图数据提取、消息序列化。

**编排层（agent/ + orchestration/）**：Agent 工厂装配、会话上下文管理、分层编排、工具元数据管理。

**协议层（mcp/）**：MCP Server、工具注册、适配器工厂、拦截器。

**领域层（tools/）**：17 个核心工具，按 search / planning / rendering / utility 四个子领域组织。

**基础设施层（storage/ + utils/ + config.py）**：三层记忆、Prompt 模板引擎、配置管理。

依赖方向：`api → agent → orchestration → tools` / `mcp → tools`。基础设施层被所有上层依赖。

---

### Q34 模型和 Agent 的区别到底是什么？

**回答**：在本项目中，模型是一个**函数**——接收消息列表，返回文本。Agent 是一个**系统**——在模型外面包了一层循环：模型输出 tool_calls → 执行工具 → 结果追加到上下文 → 模型再次推理 → ... 直到输出纯文本。

项目中的体现：`_build_llm()` 创建的是一个 ChatOpenAI 实例（模型），`create_react_agent(model=llm, tools=tools)` 把它包装成一个能自主调用工具的 Agent。模型本身不知道工具的存在，是 Agent 框架赋予了它这个能力。

---

### Q35 Agent 系统里模型和系统代码的职责边界怎么划？

**回答**：

| 模型负责 | 系统代码负责 |
|---|---|
| 决定调用哪个工具 | 工具 Schema 校验、工具执行 |
| 推断工具参数 | 结果持久化（ArtifactStore） |
| 生成回复文本 | 地图数据提取、A2UI 卡片生成 |
| — | 上下文压缩策略（何时压缩、阈值判断） |
| — | 重试/超时/熔断 |

**核心原则**：模型只负责理解和决策，系统代码负责执行和保障。

---

### Q36 为什么很多团队做 Workflow + Agent 混合架构？

**回答**：项目的分层编排就是混合架构的一个实例。

不同环节对确定性的要求不同：前置校验（检查用户是否说了目的地）可以完全确定——用规则 if-else。工具调用顺序（先搜景点还是先查天气）无法穷举——交给 LLM 自主决策。

分层编排更进一步：每层有工具白名单（确定性），但层内 LLM 自由选择调用哪个工具（灵活性）。research 层必须触发搜索工具（确定性约束），但具体搜什么、用什么关键词由 LLM 决定（灵活性）。

---

### Q37 什么时候该做 Agent？和 Workflow 的边界在哪？

**回答**：判断标准是决策路径是否可穷举。

项目中 Workflow 做的事：前置校验——4 种缺失场景（太短/无意图/无目的地/无天数）可以穷举，用 if-else 处理。

Agent 做的事：工具调用顺序和参数选择——用户可能先说"推荐酒店"再补"还要景点"，也可能直接说"成都 3 天亲子游"，LLM 自主决定搜索策略。

混合做的事：兜底策略——LLM 忘调渲染工具时，后端用确定性代码补救。

---

### Q38 开发 Agent 时踩过什么坑？

**回答**：5 个关键坑：

1. **DeepSeek 不接受 list 类型 content**：需要两层拍平——发送前在 `DeepSeekChatOpenAI._get_request_payload()` 中拍平，接收后在消息历史清理中再次拍平。

2. **分层编排死锁**：回滚后 LLM 做出相同决策再次失败。解决：默认关闭分层编排。

3. **ReAct 循环无上限**：没传 `recursion_limit` 时默认 25，LLM 可能循环到死。解决：显式传入 `recursion_limit=20`。

4. **reasoning_content 丢失**：DeepSeek thinking 模式要求回传 reasoning_content，但父类 ChatOpenAI 会丢弃它。解决：在子类中手动回注。

5. **MCP 信封两层反序列化**：render 工具返回 JSON string，外面又包了 MCP 信封 `{artifact_id, result}`。需要两次 `json.loads()`。

---

### Q39 Agent 的成本怎么控制？

**回答**：

1. `recursion_limit=20`——限制单次请求的 LLM 调用次数
2. L1 压缩——消息超过 24 条时压缩，减少每轮的 token 消耗
3. 硬阈值关闭 L2——消息超过 60 条时不再注入工具结果快照
4. 单 worker 部署——每个容器只运行一个 FastAPI worker（因为 MCP Server 绑定端口 8002）
5. L2 快照注入——让 LLM 知道"已经搜过了"，减少重复工具调用
