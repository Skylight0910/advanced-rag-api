# Week 2 Advanced RAG Summary

## 本周主题

本周围绕 Advanced RAG 展开，从一个基础 Naive RAG API 逐步扩展到支持 Query Transformation、Rerank、Evaluation、Milvus 和文档处理的实验系统。

本周核心问题不是“能不能跑通 RAG”，而是：

```text
如何让 RAG 更准、更可评估、更接近真实生产系统？
```

## 本周完成内容

| Day | 主题 | 主要成果 |
|---|---|---|
| Day 8 | Query Transformation | 新增 `/ask_multi_query` 和 `/ask_hyde` |
| Day 9 | Rerank | 新增 `/ask_rerank`，实现 Chroma top-k + CrossEncoder rerank |
| Day 10 | RAG Evaluation | 构建 `eval_questions.json`，自动调用 4 个 RAG 接口生成 `eval_results.json` |
| Day 11 | Manual Evaluation | 构建 `eval_scores.csv`，对代表性问题进行人工评分 |
| Day 12 | Milvus | 启动 Milvus standalone，跑通 demo，并新增 `/ask_milvus`、`/ask_milvus_rerank` |
| Day 13 | Document Processing | 新增 `ingest_docs.py`，支持 Markdown/TXT 文档导入 Chroma |
| Day 14 | Project Integration | 整理 README 和本周总结，形成可展示项目 |

## 当前接口能力

| 接口 | 检索策略 | 说明 |
|---|---|---|
| `/ask` | Chroma naive search | 直接使用原始 query 检索 |
| `/ask_multi_query` | LLM query rewrite + Chroma | 将问题改写成多个 query 后检索 |
| `/ask_hyde` | LLM hypothetical doc + Chroma | 先生成 hypothetical document 再检索 |
| `/ask_rerank` | Chroma candidates + CrossEncoder rerank | 先召回，再精排 |
| `/ask_milvus` | Milvus naive search | 使用 Milvus 作为向量数据库 |
| `/ask_milvus_rerank` | Milvus candidates + CrossEncoder rerank | Milvus 召回，reranker 精排 |

## 核心实验结论

### Naive RAG

优点：

- 实现简单
- 速度快
- 成本低
- 适合作为 baseline

局限：

- 对模糊问题、短问题、决策类问题较弱
- 召回结果高度依赖原始 query 的表达

### Multi-Query

优点：

- 能从多个角度扩展问题
- 对概念对比和决策类问题效果较好
- 答案通常更完整

局限：

- 需要额外调用 LLM
- 延迟增加明显
- 可能引入无关查询和噪声

### HyDE

优点：

- 可以增强短 query 的语义表达
- 对信息不足的问题可能有帮助

局限：

- 不稳定
- hypothetical document 如果偏离知识库，可能召回不到关键 chunk
- 本周实验中 q8 出现回答“不知道”的情况

### Rerank

优点：

- 能改善候选 chunk 排序
- 在效果和耗时之间较平衡
- 比 Multi-Query 和 HyDE 更适合作为默认增强策略

局限：

- 需要额外加载 reranker 模型
- candidate_k 越大，推理成本越高

### Milvus

优点：

- 更接近生产级向量数据库
- 适合大规模向量检索
- 可以替换 Chroma 的检索层

局限：

- 部署复杂度明显高于 Chroma
- 依赖 etcd、MinIO、Milvus standalone
- 本地 Docker 镜像拉取和网络配置可能成为阻塞点

### Milvus + Rerank

本周实验显示，Milvus 直接 top-k 检索时，如果 top-k 较小，可能漏掉关键 chunk。提高 top-k 可以提升召回，但会引入更多噪声。

较合理的生产级路线是：

```text
Milvus 负责大规模候选召回
Reranker 负责精排和过滤噪声
LLM 基于精排后的上下文生成答案
```

## 评估结果总结

Day 11 对 q1、q7、q8 三个代表性问题进行了人工评分：

| 接口 | 样本数 | retrieval_relevance | answer_correctness | groundedness | completeness | overall |
|---|---:|---:|---:|---:|---:|---:|
| `/ask` | 3 | 3.67 | 4.33 | 5.00 | 3.33 | 4.08 |
| `/ask_multi_query` | 3 | 4.33 | 5.00 | 5.00 | 5.00 | 4.83 |
| `/ask_hyde` | 3 | 3.67 | 3.67 | 5.00 | 2.67 | 3.75 |
| `/ask_rerank` | 3 | 4.67 | 5.00 | 5.00 | 4.33 | 4.75 |

观察：

- `/ask_multi_query` 在代表性样本中综合得分最高，但耗时更高
- `/ask_rerank` 综合得分接近 Multi-Query，且延迟更低
- `/ask_hyde` 表现不稳定，不适合作为默认策略
- Advanced RAG 不一定总是优于 Naive RAG，需要结合问题类型和成本判断

## 文档处理结论

Day 13 新增了 `ingest_docs.py`，完成了 Markdown/TXT 到 Chroma 的 ingest pipeline。

流程：

```text
docs/
-> load .md / .txt
-> clean text
-> split chunks
-> add metadata
-> write to Chroma
-> /ask 检索验证
```

测试结果：

新增 `docs/sample.md` 后，`/ask` 能够正确回答：

```text
如果需要更可控的多步骤 Agent，应该优先考虑 LangGraph。
```

说明：

```text
docs/sample.md -> ingest_docs.py -> Chroma -> /ask
```

链路已经跑通。

## 关键工程问题与处理

### 代理与 Docker

问题：

- Ubuntu shell 可以访问代理
- Docker daemon 配置代理后，仍然无法稳定拉取 Docker Hub 镜像

处理：

- 给 Docker 和 containerd 配置 systemd proxy
- 使用 `curl -x` 验证 Docker Hub registry 可达
- 最终使用 DaoCloud 镜像源拉取 MinIO 和 Milvus 镜像

结论：

```text
真实工程中，模型和代码之外，网络、代理、镜像源、Docker daemon 都可能成为关键阻塞点。
```

### Milvus search 返回空结果

问题：

- `milvus_demo.py` 初次 insert 后 search 返回空

原因：

- 插入后没有执行 `flush` 和 `load_collection`

处理：

```python
client.flush(collection_name=COLLECTION_NAME)
client.load_collection(collection_name=COLLECTION_NAME)
```

### Milvus top-k 不足

问题：

- `/ask_milvus` top 5 检索时没有召回关键 chunk，答案偏保守

处理：

- 将 Milvus candidate_k 提高到 10
- 新增 `/ask_milvus_rerank`

结论：

```text
向量数据库替换不自动提升答案质量，top-k、rerank 和上下文构建仍然关键。
```

## 当前项目亮点

- 实现了从 Naive RAG 到 Advanced RAG 的完整演进
- 支持多种检索策略横向对比
- 引入 Milvus，具备生产级向量数据库雏形
- 构建了固定测试集和人工评分流程
- 支持本地文档 ingest
- 保留 `retrieved_chunks`、`scores` 等调试字段
- 记录了真实排障过程，而不仅是理想化 demo

## 面试讲法

可以这样概括项目：

```text
我做了一个 Advanced RAG API 项目，从最基础的 Chroma 向量检索开始，逐步加入 Multi-Query、HyDE、Rerank、Milvus 和文档 ingest pipeline。

为了比较不同策略的效果，我构建了固定测试集和人工评分表，发现 Advanced RAG 并不总是更好。Multi-Query 答案更完整但成本更高，HyDE 不稳定，Rerank 在效果和延迟之间更平衡。

在接入 Milvus 时，我也遇到了 Docker 镜像拉取、containerd 代理、Milvus flush/load 等真实工程问题。最终实现了 /ask_milvus 和 /ask_milvus_rerank，并验证 Milvus + Rerank 更适合作为生产级两阶段检索方案。
```

如果面试官追问“你最大的收获是什么”，可以回答：

```text
RAG 的效果不是靠换一个更强的模型或向量库就能解决的。真正影响效果的是文档处理、chunk 策略、召回 top-k、rerank、评估体系和工程稳定性。Advanced RAG 的价值在于针对具体问题选择合适策略，而不是盲目堆复杂度。
```

## 后续计划

- 支持 PDF / HTML / Word 文档处理
- 在 API 返回中展示 metadata/source
- 将 ingest pipeline 同步写入 Chroma 和 Milvus
- 增加 `/ask_milvus_multi_query`
- 整理 Docker Compose，实现 API + Milvus 一键启动
- 学习 Agent Tool Calling，并在当前 RAG 项目上叠加工具调用能力
