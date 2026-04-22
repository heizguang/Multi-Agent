# 多智能体智能数据查询系统 v3.0

> v3.0 新增：联网搜索子智能体（Tavily）、搜索+SQL 联合对比分析、流式来源展示、6 种意图路由。

基于 LangGraph 的**一主三从**多智能体架构，支持自然语言数据查询、深度分析、**联网搜索**、数据可视化和长短期记忆。

https://github.com/user-attachments/assets/597693d0-df09-4198-93bd-242497f23e09

## 架构设计

```
MultiAgentSystem (agent.py)
    └── MasterAgent（主智能体 - 路由 / 协调 / 记忆）
            ├── SQLQueryAgent     （子智能体1 - NL2SQL + 自动纠错）
            ├── DataAnalysisAgent （子智能体2 - 数据分析 + ECharts 可视化）
            └── WebSearchAgent    （子智能体3 - Tavily 联网搜索）⭐NEW
```

### 意图路由（6 种）

| 意图 | 触发场景 | 调用链路 |
|------|---------|---------|
| `simple_answer` | 简单问候/闲聊 | 主智能体直接回答 |
| `sql_only` | 纯数据查询 | SQLQueryAgent |
| `analysis_only` | 分析已有结果 | DataAnalysisAgent |
| `sql_and_analysis` | 查询 + 深度分析 | SQL → Analysis |
| `web_search` | 联网信息检索 ⭐NEW | WebSearchAgent |
| `search_and_sql` | 内外部数据对比 ⭐NEW | SQL + WebSearchAgent |

## 核心功能

### 1. 联网搜索（WebSearchAgent）⭐NEW

- **纯搜索模式**：调用 Tavily API 检索互联网信息，LLM 综合多来源内容生成回答，附带可信来源 URL 列表
- **搜索+SQL 联合对比**：同时查询内部数据库与互联网，LLM 从两个维度对比分析，适用于"内部薪资 vs 行业水平"等场景
- **优雅降级**：未配置 `TAVILY_API_KEY` 时 `available=False`，系统正常运行，搜索功能提示配置即可
- **格式兼容**：适配 Tavily 新旧 API 返回格式（tuple / list / str）

### 2. NL2SQL 查询（SQLQueryAgent）

- Few-shot 提示词引导 LLM 生成 SQL
- **Reflection 自动纠错**：执行失败时将错误信息反馈 LLM 重新生成，最多重试 3 次
- 通过 **MCP 协议**调用独立 SQL 服务器执行查询，支持扩展多数据源

### 3. 数据分析与可视化（DataAnalysisAgent）

- 文字洞察：自动统计数值字段（最小/最大/平均），生成分析报告
- **ECharts 可视化**：LLM 自动选择图表类型（bar / line / pie）并生成配置，前端直接渲染

### 4. 双层记忆系统

- **短期记忆（MemorySaver）**
  - 会话内对话历史保留
  - 智能压缩：消息 > 10 条或 token > 1000 时 LLM 自动总结
  - 支持引用历史查询结果
- **长期记忆（LongTermMemory）**
  - 跨会话持久化用户偏好和知识（SQLite）
  - 自动提取：对话 ≥ 6 条消息时触发
  - 个性化上下文：意图识别时注入用户历史
  - 存储结构：`users`、`user_preferences`、`user_knowledge` 三张表

### 5. 流式输出

支持 SSE（Server-Sent Events）实时推送，前端接收 7 种事件：

| 事件类型 | 说明 |
|---------|------|
| `status` | 当前处理步骤描述 |
| `intent` | 识别出的意图标签 |
| `sql` | 生成的 SQL 语句 |
| `sources` | 搜索来源 URL 列表 ⭐NEW |
| `chart` | ECharts 图表配置 JSON |
| `chunk` | 回答文字流片段 |
| `done` | 完成信号 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置 API 密钥

```bash
# Windows
set OPENAI_BASE_URL=https://www.v2code.cc
set OPENAI_API_KEY=your_openai_key
set OPENAI_MODEL=gpt-5.4
# or
$env:OPENAI_BASE_URL = "https://www.v2code.cc"
$env:OPENAI_API_KEY = "your_openai_key"
$env:OPENAI_MODEL = "gpt-5.4"

# Linux/Mac
export OPENAI_BASE_URL=https://www.v2code.cc
export OPENAI_API_KEY=your_openai_key
export OPENAI_MODEL=gpt-5.4
```

**（可选）配置 Tavily 联网搜索**

```bash
# Windows
set TAVILY_API_KEY=your_tavily_key
# or
$env:TAVILY_API_KEY = "your_tavily_key"
.\start_web.bat

# Linux/Mac
export TAVILY_API_KEY=your_tavily_key
```

> Tavily API Key 申请：[https://app.tavily.com](https://app.tavily.com)，免费套餐每月 1000 次请求。
> 也可在 `config/config.yaml` 的 `search.tavily_api_key` 字段直接填写。

### 3. 初始化数据库

```bash
cd intelligent_data_query/data
python init_db.py              # 初始化业务数据库
python init_memory_db.py       # 初始化长期记忆数据库
```

### 4. 启动 Web 前端

```bash
# Windows
cd intelligent_data_query
start_web.bat

# Linux/Mac
cd intelligent_data_query
chmod +x start_web.sh
./start_web.sh
```

浏览器访问：`http://localhost:5000`

### 5. 命令行模式

```bash
cd intelligent_data_query
python agent.py
```

系统提示输入用户 ID，同一 `user_id` 可跨会话保留个人偏好。

**特殊命令**：
- `new` — 开始新会话（清空短期记忆，保留长期记忆）
- `info` — 查看当前用户信息和偏好
- `exit/quit` — 退出系统

## 配置说明

编辑 `config/config.yaml`：

```yaml
llm:
  provider: "dashscope"
  model: "${OPENAI_MODEL}"
  base_url: "${OPENAI_BASE_URL}"
  api_key: "${OPENAI_API_KEY}"
  use_responses_api: false
  temperature: 0.1
  max_tokens: 2048

database:
  path: "./data/company.db"

nl2sql:
  num_examples: 3          # Few-shot 示例数量

memory:
  long_term_db: "./data/long_term_memory.db"
  short_term_max_tokens: 1000
  compression_threshold: 10
  auto_extract_knowledge: true

# 联网搜索配置 ⭐NEW
search:
  tavily_api_key: ""        # 留空则从环境变量读取
  max_results: 5            # 每次搜索返回最大结果数
```

## REST API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 返回前端页面 |
| `/api/login` | POST | 用户登录，返回长期记忆偏好和知识 |
| `/api/query` | POST | 阻塞式查询 |
| `/api/query_stream` | POST | 流式 SSE 查询（推荐） |
| `/api/new_session` | POST | 新建会话 |
| `/api/user_info` | POST | 获取用户信息和知识列表 |
| `/api/health` | GET | 健康检查，返回 `web_search` 可用状态 ⭐NEW |

## 测试问题

**简单问答**
- 你好，你能帮我做什么？

**数据查询**
- 公司总共有多少个部门？每个部门分别在哪个城市？
- 研发部有多少名员工？他们的职位分布是怎样的？
- 哪些员工的基本工资超过 30000 元？

**查询 + 分析**
- 对比一下公司研发部、产品部和设计部的平均薪资水平
- 找出薪资最高的 10 名员工，分析他们的职位和部门分布特征

**仅分析**
- 帮我分析一下上一次查询的数据

**联网搜索** ⭐NEW
- 2025 年互联网行业软件工程师的平均薪资是多少？
- 目前 AI 大模型领域的就业趋势如何？

**搜索 + SQL 联合对比** ⭐NEW
- 我们公司研发部的薪资水平和行业平均水平相比怎么样？
- 我们公司的薪资结构在同行业中处于什么水平？

## 目录结构

```
intelligent_data_query/
├── agents/                       # 智能体模块
│   ├── __init__.py
│   ├── master_agent.py          # 主智能体（路由 / 记忆 / 汇总）
│   ├── sql_agent.py             # SQL 查询子智能体（含自动纠错）
│   ├── analysis_agent.py        # 数据分析子智能体（含 ECharts）
│   └── search_agent.py          # 联网搜索子智能体（Tavily）⭐NEW
├── memory/                       # 记忆模块
│   ├── long_term_memory.py      # 长期记忆管理器（SQLite）
│   └── memory_extractor.py      # 记忆提取器（LLM 自动提取）
├── config/
│   └── config.yaml              # 配置文件（含搜索配置）
├── data/
│   ├── company.db               # 业务数据库
│   ├── long_term_memory.db      # 长期记忆数据库
│   ├── init_db.py               # 业务数据库初始化
│   └── init_memory_db.py        # 记忆数据库初始化
├── static/                       # Web 前端（v3.0）
│   ├── index.html               # 主页面（含搜索来源展示）
│   ├── style.css                # 样式（蓝紫渐变主题）
│   └── app.js                   # 前端逻辑（SSE 流式 / ECharts / 来源）
├── agent.py                      # 主入口（MultiAgentSystem 类）
├── app.py                        # Flask Web API 服务
├── prompts.py                    # 提示词定义（含搜索综合提示词）
├── mcp_sql_server.py             # MCP SQL 服务器
├── start_web.bat                 # Windows 启动脚本
├── start_web.sh                  # Linux/Mac 启动脚本
└── video/
    └── demo.mp4                  # 演示视频
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 工作流编排 | LangGraph |
| LLM 框架 | LangChain |
| 大语言模型 | 通义千问（qwen-turbo） |
| 联网搜索 | Tavily（langchain-tavily）⭐NEW |
| 数据库协议 | MCP（Model Context Protocol） |
| 数据存储 | SQLite |
| Web 框架 | Flask + Flask-CORS |
| 前端可视化 | ECharts、marked.js、highlight.js |
| 终端美化 | Rich |

## 已完成功能 ✅

- ~~**一主两从多智能体**~~：意图识别、NL2SQL、数据分析 ✅ (v1.0)
- ~~**双层记忆系统**~~：短期压缩 + 长期持久化 ✅ (v2.0)
- ~~**用户登录与会话管理**~~：多用户、跨会话偏好 ✅ (v2.0)
- ~~**Web 前端**~~：Markdown 渲染、ECharts 可视化、即时读取长期记忆 ✅ (v2.1)
- ~~**流式输出**~~：SSE 实时推送、打字机效果 ✅ (v2.1)
- ~~**联网搜索子智能体**~~：Tavily 集成、优雅降级 ✅ (v3.0)
- ~~**搜索+SQL 联合对比**~~：内外部数据双维度分析 ✅ (v3.0)
- ~~**来源 URL 展示**~~：搜索结果可溯源 ✅ (v3.0)

## 下一步计划

### 智能体扩展
- **报表生成子智能体**：自动生成 PDF/Excel 报表
- **异常检测子智能体**：主动发现数据异常并告警
- **多智能体并行**：对独立任务实现并行调用优化

### 记忆系统升级
- **向量检索**：使用 ChromaDB 替代简单的 LIKE 匹配
- **记忆衰减机制**：根据时间和访问频率调整知识置信度
- **全文搜索**：为 `user_knowledge` 添加 FTS5 全文索引

### 数据源扩展
- **多数据库支持**：MySQL、PostgreSQL、ClickHouse
- **Schema 智能检索**：大型数据库仅获取相关表结构

## 注意事项

- 必须设置 `OPENAI_API_KEY` 环境变量（推荐放在 `.env`）
- 使用中转时请设置 `OPENAI_BASE_URL`（例如 `https://www.v2code.cc`）
- 模型建议配置为 `OPENAI_MODEL=gpt-5.4`（可按支持列表自行更换）
- 联网搜索需额外设置 `TAVILY_API_KEY`（不配置不影响其他功能）
- 初次运行前需执行数据库初始化脚本
- 使用相同的 `user_id` 可跨会话保留个人偏好
- 长期记忆数据库建议定期备份（`data/long_term_memory.db`）

## 更新日志

### v3.0 (2026.03.10)
- ✨ 新增：联网搜索子智能体 `WebSearchAgent`（基于 Tavily）
- ✨ 新增：`web_search` 和 `search_and_sql` 两种意图路由（共 6 种）
- ✨ 新增：搜索+SQL 联合对比分析模式（内外部数据双维度）
- ✨ 新增：搜索来源 URL 在前端流式展示（`sources` SSE 事件）
- ✨ 新增：`/api/health` 返回 `web_search` 可用状态
- ✨ 新增：搜索综合提示词 `get_search_synthesis_prompt` / `get_search_and_sql_prompt`
- 🔧 优化：搜索智能体不可用时自动降级为 `simple_answer` 并友好提示
- 🔧 优化：前端意图标签新增 🌐 图标支持

### v2.1 (2025.11.05)
- ✨ 新增：内置 Web 前端（`static/`），支持 Markdown 渲染与代码高亮
- ✨ 新增：REST API（`/api/login`、`/api/query`、`/api/new_session`、`/api/user_info`、`/api/health`）
- ✨ 新增：登录后直接读取 `long_term_memory.db`，即时展示用户偏好/知识
- ✨ 新增：启动脚本 `start_web.bat` / `start_web.sh`
- 🎬 新增：项目演示视频 `video/demo.mp4`
- 🔧 优化：聊天滚动条与输入框体验

### v2.0 (2025.10.21)
- ✨ 新增：长期记忆系统（用户偏好、知识跨会话持久化）
- ✨ 新增：短期记忆智能压缩（LLM 自动总结）
- ✨ 新增：用户登录和会话管理系统
- ✨ 新增：自动记忆提取（从对话中提取偏好和知识）
- 🔧 优化：意图识别注入用户历史上下文

### v1.0 (2025.10.16)
- 🎉 初始版本
- ✨ 一主两从多智能体架构
- ✨ 意图识别和智能路由（4 种）
- ✨ NL2SQL 查询和数据分析
- ✨ 短期会话记忆（MemorySaver）
