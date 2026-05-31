# Advanced RAG API

一个基于 FastAPI、LangChain、Chroma、Milvus 和 DeepSeek 的 Advanced RAG 实验项目。

项目从 Naive RAG 开始，逐步实现 Query Transformation、Rerank、Hybrid Search、Milvus 接入、文档导入和评估流程，用于比较不同 RAG 策略在召回能力、回答质量、延迟和工程复杂度上的差异。

## 核心功能

- Naive RAG：基于 Chroma 的基础向量检索问答
- Multi-Query RAG：使用 LLM 将用户问题改写为多个检索 query
- HyDE RAG：使用 LLM 生成 hypothetical document 增强检索
- Rerank RAG：先召回候选 chunks，再使用 CrossEncoder 精排
- Hybrid Search：BM25 关键词检索 + Embedding 语义检索，使用 RRF 融合排名
- Hybrid + Rerank：混合召回候选 chunks，再使用 CrossEncoder 精排
- Milvus RAG：使用 Milvus 作为向量数据库进行检索
- Milvus + Rerank：Milvus 召回候选 chunks，再使用 CrossEncoder 精排
- RAG Evaluation：固定测试集、自动调用脚本和人工评分表
- Document Ingest：加载 Markdown、TXT 文档，切分后写入 Chroma

## 技术栈

| 类别 | 技术 |
|---|---|
| API 框架 | FastAPI |
| LLM 编排 | LangChain / LCEL |
| LLM | DeepSeek |
| Embedding | HuggingFaceEmbeddings |
| 本地向量库 | Chroma |
| 向量数据库 | Milvus |
| 关键词检索 | BM25 / rank-bm25 / jieba |
| Reranker | sentence-transformers CrossEncoder |
| 文档切分 | RecursiveCharacterTextSplitter |
| 部署 | Docker / Docker Compose |

## 系统架构

```text
用户问题
  |
  v
FastAPI
  |
  +-- /ask -----------------> Chroma search -> context -> LLM
  |
  +-- /ask_multi_query -----> LLM rewrite queries -> Chroma search -> context -> LLM
  |
  +-- /ask_hyde ------------> LLM hypothetical doc -> Chroma search -> context -> LLM
  |
  +-- /ask_rerank ----------> Chroma top-k -> CrossEncoder rerank -> context -> LLM
  |
  +-- /ask_hybrid ----------> BM25 + Embedding -> RRF fusion -> context -> LLM
  |
  +-- /ask_hybrid_rerank ---> BM25 + Embedding -> RRF fusion -> CrossEncoder rerank -> context -> LLM
  |
  +-- /ask_milvus ----------> Milvus search -> context -> LLM
  |
  +-- /ask_milvus_rerank ---> Milvus top-k -> CrossEncoder rerank -> context -> LLM
```

## Hybrid Search 设计

Hybrid Search 使用两路召回：

```text
BM25 关键词检索 Top 10
        +
Embedding 语义检索 Top 10
        |
        v
RRF 排名融合与去重
        |
        v
可选 CrossEncoder Rerank
        |
        v
Top chunks -> LLM
```

- BM25 擅长精确关键词、专有名词和版本号。
- Embedding 擅长同义表达和语义相似问题。
- 中文文本使用 `jieba` 分词后再建立 BM25 索引。
- BM25 分数和向量相似度量纲不同，因此使用 RRF（Reciprocal Rank Fusion）按排名融合，而不是直接相加原始分数。

## 项目结构

```text
p1-project/
├── app.py
├── ingest_docs.py
├── milvus_demo.py
├── agent_llm_demo.py
├── eval_manual.py
├── eval_questions.json
├── eval_results.json
├── eval_scores.csv
├── chroma_db/
├── docs/
│   └── sample.md
├── milvus/
│   └── docker-compose.yml
├── experiments/
│   ├── day8-query-transformation.md
│   ├── day9-rerank.md
│   ├── day9-hybrid-search.md
│   ├── day10-rag-evaluation.md
│   ├── day11-manual-evaluation.md
│   ├── day12-milvus.md
│   ├── day13-document-processing.md
│   └── day15-tool-calling-agent.md
└── README.md
```

## 环境变量

在 `.env` 中配置 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=your_api_key
```

## 安装依赖

```bash
cd ~/code/p1-project
source .venv/bin/activate
pip install rank-bm25 jieba
```

## 启动服务

当前 `app.py` 会在启动时连接 Milvus，因此需要先启动 Milvus：

```bash
cd ~/code/p1-project/milvus
sudo docker-compose up -d
sudo docker-compose ps
```

再启动 API：

```bash
cd ~/code/p1-project
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

访问 Swagger 文档：

```text
http://localhost:8000/docs
```

Milvus 默认端口：

| 端口 | 用途 |
|---|---|
| `19530` | Milvus gRPC 服务 |
| `9091` | Milvus health / management |
| `9000`、`9001` | MinIO |

## 文档导入

将 Markdown 或 TXT 文档放入 `docs/`：

```text
docs/sample.md
```

运行：

```bash
cd ~/code/p1-project
source .venv/bin/activate
python ingest_docs.py
```

当前 ingest pipeline 支持：

- `.md`
- `.txt`
- metadata：`source`、`file_name`、`file_type`、`chunk_id`、`chunk_size`、`chunk_overlap`
- chunk 参数：`CHUNK_SIZE = 500`，`CHUNK_OVERLAP = 100`

注意：BM25 索引在 API 启动时从 Chroma 构建。重新导入文档后，需要重启 API。

## 接口列表

请求体统一使用：

```json
{"query": "你的问题"}
```

| 接口 | 说明 |
|---|---|
| `GET /` | 健康检查和接口列表 |
| `POST /ask` | Chroma 向量检索 baseline |
| `POST /ask_multi_query` | LLM 改写多个 query 后合并召回 |
| `POST /ask_hyde` | 使用 hypothetical document 检索 |
| `POST /ask_rerank` | Chroma 候选召回 + CrossEncoder 精排 |
| `POST /ask_hybrid` | BM25 + Embedding + RRF |
| `POST /ask_hybrid_rerank` | BM25 + Embedding + RRF + CrossEncoder |
| `POST /ask_milvus` | Milvus 向量检索 |
| `POST /ask_milvus_rerank` | Milvus 候选召回 + CrossEncoder 精排 |

示例：

```bash
curl http://localhost:8000/ask_hybrid_rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"如果我想要一个更可控的多步骤 Agent，应该选什么？"}'
```

## RAG Evaluation

项目包含一套轻量评估流程：

- `eval_questions.json`：固定测试问题集
- `eval_manual.py`：自动调用不同 RAG 接口
- `eval_results.json`：保存接口返回和耗时
- `eval_scores.csv`：人工评分

运行：

```bash
cd ~/code/p1-project
source .venv/bin/activate
python eval_manual.py
```

人工评分维度：

| 维度 | 说明 |
|---|---|
| `retrieval_relevance` | retrieved chunks 是否与问题相关 |
| `answer_correctness` | answer 是否正确 |
| `groundedness` | answer 是否基于 retrieved chunks |
| `completeness` | 是否覆盖 expected points |
| `latency` | 响应耗时 |

### 延迟测试结果

固定测试集包含 8 个问题，每个接口调用 8 次。

| 接口 | 样本数 | 平均耗时 |
|---|---:|---:|
| `/ask` | 8 | 5.258s |
| `/ask_multi_query` | 8 | 9.855s |
| `/ask_hyde` | 8 | 11.687s |
| `/ask_rerank` | 8 | 8.203s |
| `/ask_hybrid` | 8 | 4.355s |
| `/ask_hybrid_rerank` | 8 | 6.960s |

`/ask_hybrid` 本轮比 `/ask` 更快，但这不代表 Hybrid Search 天然更快。两者都只调用一次 LLM，小样本平均值会受到模型接口响应波动影响。

### 人工评分结果

| 接口 | 样本数 | retrieval_relevance | answer_correctness | groundedness | completeness | overall |
|---|---:|---:|---:|---:|---:|---:|
| `/ask` | 3 | 3.67 | 4.33 | 5.00 | 3.33 | 4.08 |
| `/ask_multi_query` | 3 | 4.33 | 5.00 | 5.00 | 5.00 | 4.83 |
| `/ask_hyde` | 3 | 3.67 | 3.67 | 5.00 | 2.67 | 3.75 |
| `/ask_rerank` | 3 | 4.67 | 5.00 | 5.00 | 4.33 | 4.75 |

## 关键结论

- Naive RAG 简单且易于调试，适合作为 baseline。
- Multi-Query 可以扩大召回范围，但会增加额外 LLM 调用和延迟。
- HyDE 对部分短 query 有帮助，但可能发生语义漂移。
- Rerank 在候选召回后改善排序，适合用于精排。
- Hybrid Search 通过 BM25 与 Embedding 互补召回，并使用 RRF 避免直接混合不同量纲的分数。
- Hybrid + Rerank 适合作为默认检索方案：先扩大候选覆盖，再进行精排。
- Milvus 提升的是索引管理、扩展性和并发能力，不会自动提高答案质量。
- 文档处理质量会显著影响 RAG 效果，metadata 和 chunk 策略值得继续优化。

## 项目亮点

- 从 Naive RAG 逐步演进到 Hybrid Search + Rerank
- 支持多种检索策略横向对比
- 使用 Chroma 快速实验，并接入 Milvus 验证工程扩展方案
- 构建固定测试集、自动调用脚本和人工评分流程
- 保留 retrieved chunks、RRF scores 和 rerank scores，便于分析答案依据
- 记录 Docker 代理、Milvus 部署和 localhost 代理绕过等真实排障过程

## 已知限制

- 当前 Milvus 未启动时，API 会启动失败；后续可改为可降级模式。
- BM25 索引在 API 启动时构建；导入新文档后需要重启 API。
- 当前 ingest pipeline 仅支持 Markdown 和 TXT。
- 当前评估采用固定测试集与人工评分，尚未接入 RAGAs 等自动化评估框架。
- 当前知识库规模较小，延迟测试用于策略对比，不代表生产环境性能。

## 后续方向

当前模板已经完成。后续优先采用问题驱动方式继续迭代：

- 构造专有名词、版本号和同义表达场景，验证 Hybrid Search
- 调整 chunk size 与 overlap，观察长文档问答效果
- 增加 metadata filter，理解多用户知识库隔离
- 支持 PDF、HTML、Word 等复杂文档
- 将文档导入同步写入 Chroma 和 Milvus
- 为 Milvus 增加可降级初始化
- 在测试集扩大后接入自动化评估框架
