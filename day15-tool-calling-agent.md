# Day 15 Tool Calling Agent

## 背景

前面两周主要围绕 RAG 系统展开，包括 Naive RAG、Query Transformation、Rerank、Evaluation、Milvus 和文档 ingest pipeline。

Day 15 开始进入 Agent 阶段。和 RAG 不同，Agent 不只是检索知识后回答，而是需要根据任务选择工具、调用工具、观察结果，并基于工具结果生成最终答案。

本日重点是理解 Tool Calling Agent 的最小闭环。

## 今日目标

- 理解 Agent 和 RAG 的区别
- 实现一个规则版工具调用 demo
- 实现一个 LLM 自主选择工具的 demo
- 理解 Tool Schema、Tool Selection、Tool Execution、Observation、Final Answer
- 将现有 RAG API 包装为 Agent 可调用工具

## Agent vs RAG

RAG 的核心流程：

```text
用户问题
-> 检索相关文档
-> 拼接 context
-> LLM 基于 context 回答
```

Agent 的核心流程：

```text
用户问题
-> LLM 判断是否需要工具
-> LLM 选择工具并生成参数
-> Python 执行工具
-> 工具结果作为 observation 返回给 LLM
-> LLM 生成最终答案
```

区别：

| 对比项 | RAG | Agent |
|---|---|---|
| 核心能力 | 检索知识并回答 | 选择工具、执行动作、整合结果 |
| 主要输入 | 用户问题 + 检索上下文 | 用户问题 + 工具列表 |
| 关键过程 | Retrieval + Generation | Tool Selection + Tool Execution |
| 典型用途 | 知识库问答 | 任务执行、工具编排、自动化流程 |

## 预热版：规则工具调用

首先实现了 `agent_demo.py`。

该版本使用规则判断工具调用：

```text
如果用户输入包含数学表达式 -> 调用 calculator
如果用户输入包含 Agent/RAG/LangGraph 等关键词 -> 调用 rag_search
如果用户询问项目状态 -> 调用 project_status
```

工具包括：

- `rag_search(query)`: 调用现有 RAG API
- `calculator(expression)`: 计算简单数学表达式
- `project_status()`: 返回当前项目能力清单

该版本跑通了：

```text
用户输入
-> 程序规则判断工具
-> 执行工具
-> 拼接工具结果
-> 输出答案
```

局限：

```text
工具选择由 if/else 规则决定，不是 LLM 自主判断。
```

因此它只是 Tool Calling 的概念热身，不算真正的 Agent。

## 正式版：LLM Tool Calling Agent

随后实现了 `agent_llm_demo.py`。

该版本让 LLM 根据工具 schema 自主选择工具，并输出结构化 JSON。

完整流程：

```text
用户问题
-> 给 LLM 工具列表和工具参数说明
-> LLM 输出 tool_call JSON
-> Python 解析 JSON
-> 执行对应工具
-> 得到 observation
-> 将 observation 回传给 LLM
-> LLM 输出 final answer
```

## 可用工具

### rag_search

用途：

```text
查询当前 Advanced RAG 项目知识库。
```

适合问题：

- LangChain
- LangGraph
- Deep Agents
- MCP
- RAG
- Agent

实现方式：

```text
调用 http://localhost:8000/ask_rerank
```

### calculator

用途：

```text
计算简单数学表达式。
```

示例：

```text
12*8+5
```

### project_status

用途：

```text
返回当前项目已经支持的能力和接口。
```

包括：

- `/ask`
- `/ask_multi_query`
- `/ask_hyde`
- `/ask_rerank`
- `/ask_milvus`
- `/ask_milvus_rerank`
- `ingest_docs.py`
- `eval_manual.py`
- `eval_scores.csv`

## Tool Schema

LLM 需要知道有哪些工具，以及每个工具需要什么参数。

示例：

```json
{
  "name": "rag_search",
  "description": "查询当前 Advanced RAG 项目知识库，适合回答 LangChain、LangGraph、Deep Agents、MCP、RAG、Agent 相关问题。",
  "args": {
    "query": "string，用户要查询的问题"
  }
}
```

Tool Schema 的作用：

- 限定 LLM 可调用的工具范围
- 告诉 LLM 每个工具适合什么任务
- 告诉 LLM 参数格式
- 让工具调用可以被程序解析和执行

## Tool Call JSON

LLM 被要求只输出 JSON。

调用工具时输出：

```json
{
  "action": "tool_call",
  "tool": "rag_search",
  "arguments": {
    "query": "如果我要构建一个更可控的多步骤 Agent，应该用什么？"
  }
}
```

不需要工具时输出：

```json
{
  "action": "final_answer",
  "answer": "你的回答"
}
```

## 测试问题

本次测试了三个问题：

```text
如果我要构建一个更可控的多步骤 Agent，应该用什么？
```

预期：

- LLM 选择 `rag_search`
- 工具调用 `/ask_rerank`
- 最终回答 LangGraph

```text
帮我算一下 12*8+5。
```

预期：

- LLM 选择 `calculator`
- 工具返回 `101`

```text
当前项目已经支持哪些能力？
```

预期：

- LLM 选择 `project_status`
- 返回当前项目接口和能力清单

## 运行结果

`agent_llm_demo.py` 成功跑通。

观察到：

- LLM 能根据用户问题输出 tool_call JSON
- Python 能解析 JSON 并调用对应工具
- 工具返回结果作为 observation 传回 LLM
- LLM 能基于 observation 生成最终回答

这说明 Tool Calling Agent 的最小闭环已经完成。

## 核心概念

### Tool Schema

告诉 LLM 有哪些工具，以及每个工具需要什么参数。

### Tool Selection

LLM 根据用户问题选择是否调用工具，以及调用哪个工具。

### Tool Execution

Python 根据 LLM 输出的 tool_call 执行真实函数或 API。

### Observation

工具执行结果返回给 LLM，作为后续回答依据。

### Final Answer

LLM 基于用户问题和 observation 输出最终自然语言回答。

## 今日结论

Day 15 完成了从 RAG 到 Agent 的第一步。

预热版 `agent_demo.py` 帮助理解了“工具调用”的基本概念，但工具选择由程序规则控制。

正式版 `agent_llm_demo.py` 让 LLM 根据工具 schema 自主选择工具，并通过 JSON 输出结构化 tool_call。Python 负责执行工具，LLM 再基于工具结果生成最终答案。

本日最重要的收获是：

```text
Agent 的关键不是模型自己执行工具，而是模型决定要调用什么工具，程序负责安全地执行工具，并把结果反馈给模型。
```

## 当前限制

- 目前只支持单次工具调用
- LLM 输出 JSON 仍可能格式错误，需要更强的解析和重试机制
- 工具参数缺少严格校验
- 工具执行失败时没有统一错误处理
- 还没有多轮 tool call loop
- 没有使用 OpenAI/DeepSeek 原生 function calling 协议，而是手写 JSON tool call

## 下一步

- 增加工具调用错误处理
- 增加 JSON 解析失败时的重试逻辑
- 支持多轮 tool call
- 封装统一 Tool 类
- 增加文件读取、搜索、SQL 查询等真实工具
- 将 Agent 能力整合到当前 RAG 项目中
