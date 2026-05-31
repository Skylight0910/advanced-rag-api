# Day 13 Document Processing

## 背景

前面几天主要优化了 RAG 的检索策略和向量数据库能力：

- Day 8：Query Transformation，包括 Multi-Query 和 HyDE
- Day 9：Rerank
- Day 10-11：RAG Evaluation
- Day 12：Milvus 接入

Day 13 回到 RAG 的源头：文档如何进入知识库。

真实 RAG 项目中，效果问题不一定来自模型，也可能来自文档处理阶段。例如：

- 文档没有正确加载
- 文本清洗不干净
- chunk 切分不合理
- metadata 丢失
- 文档来源无法追踪

因此本日重点是构建第一版文档 ingest pipeline，让项目可以从本地 `docs/` 目录读取文档、切分 chunk，并写入 Chroma。

## 今日目标

- 新建 `docs/` 文档目录
- 新建 `ingest_docs.py`
- 支持 `.md` 和 `.txt` 文件读取
- 使用 `RecursiveCharacterTextSplitter` 切分文档
- 将 chunks 写入 Chroma
- 保留 metadata，例如 `source`、`file_name`、`file_type`、`chunk_id`
- 通过 `/ask` 验证新文档可以被检索并用于回答

## 项目结构

本次新增内容：

```text
p1-project/
├── docs/
│   └── sample.md
├── ingest_docs.py
└── experiments/
    └── day13-document-processing.md
```

## 样例文档

本次使用 `docs/sample.md` 作为测试文档。

文档主题是 Agent 框架选择，包含以下知识点：

- 简单 LLM 应用可以优先考虑 LangChain
- 复杂、可控、有状态的多步骤 Agent 工作流可以优先考虑 LangGraph
- 开箱即用的高级 Agent 能力可以考虑 Deep Agents
- MCP 可以连接 Claude、VSCode 等工具，用于实时文档问答

## Ingest Pipeline

`ingest_docs.py` 的处理流程：

```text
读取 docs/ 目录
-> 过滤 .md / .txt 文件
-> 读取文本内容
-> 构造 LangChain Document
-> 添加 metadata
-> 使用 RecursiveCharacterTextSplitter 切分 chunks
-> 为每个 chunk 添加 chunk_id、chunk_size、chunk_overlap
-> 使用 HuggingFaceEmbeddings 生成向量
-> 写入 Chroma
```

## Chunk 参数

本次使用的参数：

```text
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
```

切分器：

```python
RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)
```

参数含义：

- `chunk_size`: 每个 chunk 的最大长度
- `chunk_overlap`: 相邻 chunk 之间保留的重叠内容
- `separators`: 优先按照段落、换行、中文句号、中文逗号等边界切分

## Metadata 设计

每个原始文档会保留以下 metadata：

| 字段 | 说明 |
|---|---|
| `source` | 文件路径 |
| `file_name` | 文件名 |
| `file_type` | 文件类型，例如 `.md`、`.txt` |

切分后，每个 chunk 额外增加：

| 字段 | 说明 |
|---|---|
| `chunk_id` | chunk 编号 |
| `chunk_size` | 当前 ingest 使用的 chunk size |
| `chunk_overlap` | 当前 ingest 使用的 overlap |

metadata 的价值是后续可以追踪答案来源，并支持更细粒度的检索、过滤和调试。

## 测试命令

运行 ingest：

```bash
python ingest_docs.py
```

启动 API：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

测试问题：

```bash
curl http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"如果需要更可控的多步骤 Agent，应该选什么？"}'
```

## 测试结果

接口返回：

```text
如果您需要更可控的多步骤 Agent，应该优先考虑 LangGraph。
```

`retrieved_chunks` 第一条命中了新增的 `docs/sample.md` 内容：

```text
# Agent 框架选择指南

如果需要快速构建一个简单的 LLM 应用，可以优先考虑 LangChain。

如果需要构建复杂、可控、有状态的多步骤 Agent 工作流，可以优先考虑 LangGraph。

如果希望获得开箱即用的高级 Agent 能力，例如自动上下文压缩、虚拟文件系统和子代理能力，可以考虑 Deep Agents。

MCP 可以将文档连接到 Claude、VSCode 等工具
```

这说明：

```text
docs/sample.md
-> ingest_docs.py
-> Chroma
-> /ask 检索
-> LLM 回答
```

整条链路已经跑通。

## 关键观察

- 文档处理质量会直接影响 RAG 检索效果。
- 当前 sample.md 内容较短，因此一个 chunk 中就包含了主要知识点。
- `/ask` 能够优先检索到新增文档，说明 Chroma 已成功写入新 chunks。
- metadata 已经在 ingest 阶段保留，但当前 API 返回中还没有展示 metadata。
- 如果后续文档变长，需要对比不同 `chunk_size` 和 `chunk_overlap` 的效果。
- 当前 Milvus collection 不会自动同步新 ingest 的 Chroma 内容，后续需要考虑同步策略。

## 当前限制

- 当前仅支持 `.md` 和 `.txt`
- 暂未支持 PDF、HTML、Word 等复杂文档格式
- 暂未做复杂清洗，例如去除导航栏、页眉页脚、重复内容
- 暂未在 API 返回中展示 source metadata
- 暂未将新增文档同步到 Milvus
- 当前 ingest 会写入 Chroma，但没有单独处理旧 collection 的清空和重建策略

## 今日结论

Day 13 完成了第一版文档处理管线。

本次实现了从本地 Markdown 文档到 Chroma 知识库的完整 ingest 流程，并通过 `/ask` 接口验证新文档可以被检索和用于回答。

本日最重要的收获是：

```text
RAG 的效果不只取决于 LLM 和检索算法，也高度依赖文档加载、清洗、切分和 metadata 设计。
```

对于真实项目，文档处理通常是 RAG 系统质量的基础。后续需要继续增强 ingest pipeline，使其支持更多文件格式，并保留更完整的来源信息。

## 下一步

- 支持 HTML 文档加载和清洗
- 支持 PDF 文档加载
- 在 API 返回中加入 metadata/source 信息
- 对比不同 `chunk_size` 和 `chunk_overlap`
- 设计 Chroma 与 Milvus 的同步策略
- Day 14 整合当前 Advanced RAG 项目
