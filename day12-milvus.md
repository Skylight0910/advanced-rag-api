# Day 12 Milvus

## 背景

当前 RAG 项目使用 Chroma 作为本地向量库。Chroma 使用简单，适合学习、原型验证和小规模本地实验。

随着文档规模变大、并发请求增加、需要更稳定的服务化部署时，本地轻量向量库可能无法满足生产环境需求。因此 Day 12 开始学习生产级向量数据库 Milvus，并理解从 Chroma 迁移到 Milvus 的基本思路。

## 今日目标

- 理解 Chroma 和 Milvus 的区别
- 理解 Milvus 的核心概念
- 使用 Docker Compose 启动 Milvus standalone
- 使用 `milvus_demo.py` 跑通最小 insert/search 实验
- 将主 RAG 项目接入 Milvus
- 新增 `/ask_milvus` 和 `/ask_milvus_rerank` 接口

## 当前 Chroma 版本流程

当前项目中的检索流程为：

```text
用户 query
-> HuggingFaceEmbeddings 生成 query embedding
-> Chroma similarity_search 检索相关文档块
-> 拼接 retrieved_chunks 为 context
-> rag_chain 调用 LLM 生成 answer
```

对应代码位置主要是：

```python
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)
docs = vectordb.similarity_search(q.query, k=RETRIEVER_K)
```

## Chroma vs Milvus

| 对比项 | Chroma | Milvus |
|---|---|---|
| 定位 | 本地轻量向量库 | 生产级向量数据库 |
| 使用复杂度 | 简单，适合快速上手 | 更复杂，需要理解服务、collection、schema、index |
| 部署方式 | 可直接本地持久化 | 通常以服务方式运行，可用 Docker 部署 |
| 适合规模 | 小规模到中等规模 | 中大型向量数据、生产环境 |
| 检索能力 | 封装简单，使用方便 | 索引类型和检索参数更丰富 |
| 运维成本 | 低 | 更高 |
| 适合场景 | 学习、demo、本地 RAG | 企业级 RAG、多用户服务、大规模文档检索 |

## Milvus 核心概念

### Collection

Collection 类似数据库中的一张表，用来存储向量和相关字段。

在 RAG 场景中，一个 collection 可以存储：

- 文档 chunk 的 id
- 文档 chunk 的文本内容
- 文档 chunk 的 embedding 向量
- 可选 metadata，例如 source、page、title

### Schema

Schema 用来定义 collection 中有哪些字段，以及字段类型。

例如：

- `id`: 主键
- `text`: 原始文本 chunk
- `embedding`: 向量字段
- `source`: 文档来源

### Index

Index 用来加速向量检索。

如果没有索引，向量搜索可能会变慢。Milvus 支持多种索引方式，适合不同规模和性能需求。

### Insert

Insert 是将文本 chunk 和对应 embedding 写入 Milvus。

在 RAG 中，通常流程是：

```text
文档 -> chunk -> embedding -> insert into Milvus
```

### Search

Search 是根据 query embedding 在 Milvus 中搜索最相似的向量。

在 RAG 中，通常流程是：

```text
用户 query -> query embedding -> Milvus search -> top-k chunks
```

## 从 Chroma 迁移到 Milvus 的思路

当前 Chroma 版本：

```text
query -> embedding_model -> Chroma similarity_search -> docs -> context -> LLM
```

迁移到 Milvus 后：

```text
query -> embedding_model -> Milvus search -> matched chunks -> context -> LLM
```

需要替换的主要是“检索层”。

也就是说，`rag_chain`、`llm`、`prompt`、FastAPI 接口本身可以基本保留，主要把：

```python
vectordb.similarity_search(...)
```

替换为：

```python
milvus_search(...)
```

## Docker 启动 Milvus

本次使用 Milvus standalone Docker Compose 启动以下服务：

- `milvus-etcd`
- `milvus-minio`
- `milvus-standalone`

启动命令：

```bash
cd ~/code/p1-project/milvus
sudo docker-compose up -d
sudo docker-compose ps
```

启动成功后，Milvus 暴露端口：

- `19530`: Milvus 服务端口
- `9091`: Milvus Web UI/管理端口
- `9000` / `9001`: MinIO 端口

## Docker 镜像拉取问题

启动 Milvus standalone 时，Docker 拉取 Docker Hub 镜像出现连接失败。

典型错误：

```text
failed to resolve reference "docker.io/minio/minio:...": connect: connection refused
```

排查过程：

- Ubuntu shell 可以通过代理访问外网
- `curl -x http://192.168.31.1:7891 https://registry-1.docker.io/v2/` 可以返回 `401`
- Docker daemon 和 containerd 都配置了 systemd 代理
- 进程环境中也能看到 `HTTP_PROXY` 和 `HTTPS_PROXY`
- 但 `docker pull` 仍然出现直连 Docker Hub 的现象

最终处理方式：

- 使用 DaoCloud 镜像源提前拉取 Docker Hub 镜像
- 将 `docker-compose.yml` 中的 MinIO 和 Milvus 镜像地址替换为 DaoCloud 镜像源

示例：

```yaml
image: docker.m.daocloud.io/minio/minio:RELEASE.2024-12-18T13-15-44Z
image: docker.m.daocloud.io/milvusdb/milvus:v2.6.17
```

替换后 Milvus standalone 成功启动。

## Milvus Demo 实验

本次通过 `milvus_demo.py` 完成最小向量检索实验。

### Demo 流程

```text
连接 Milvus
-> 创建 collection
-> 使用 HuggingFaceEmbeddings 生成 768 维向量
-> 插入 4 条测试文本
-> flush + load_collection
-> 使用 query embedding 执行向量搜索
```

测试文本包括：

- LangChain 适合快速构建 LLM 应用和 Agent
- LangGraph 适合构建复杂、可控、有状态的多步骤 Agent 工作流
- Deep Agents 提供开箱即用的 Agent 能力
- MCP 可以将文档连接到 Claude、VSCode 等工具

实验问题：

```text
如果我想要一个更可控的多步骤 Agent，应该选什么？
```

搜索结果第一条：

```text
LangGraph 适合构建复杂、可控、有状态的多步骤 Agent 工作流。
```

说明 Milvus 可以正常完成语义向量检索。

### Demo 调试记录

初次执行 search 时返回空结果。原因是插入数据后没有执行 `flush` 和 `load_collection`。

补充以下步骤后搜索正常：

```python
client.flush(collection_name=COLLECTION_NAME)
client.load_collection(collection_name=COLLECTION_NAME)
```

## 主 RAG 项目接入 Milvus

在 Milvus demo 跑通后，主 RAG 项目采用“新增接口，不替换原接口”的方式接入 Milvus。

保留原有接口：

- `POST /ask`
- `POST /ask_multi_query`
- `POST /ask_hyde`
- `POST /ask_rerank`

新增接口：

- `POST /ask_milvus`
- `POST /ask_milvus_rerank`

这样可以继续对比 Chroma 和 Milvus 的检索效果，避免直接替换导致 baseline 丢失。

## Milvus 数据初始化

本次采用从现有 Chroma 向量库读取 documents，再写入 Milvus 的方式。

流程：

```text
Chroma get documents
-> 逐条生成 embedding
-> insert into Milvus
-> flush
-> load_collection
```

主要函数：

```python
def init_milvus_from_chroma():
    chroma_data = vectordb.get(include=["documents"])
    documents = chroma_data.get("documents", [])
    ...
    milvus_client.insert(...)
    milvus_client.flush(...)
    milvus_client.load_collection(...)
```

第一次构建时使用：

```python
REBUILD_MILVUS_ON_START = True
```

跑通后应改回：

```python
REBUILD_MILVUS_ON_START = False
```

避免每次启动 API 都重建 Milvus collection。

## `/ask_milvus`

`/ask_milvus` 使用 Milvus 直接进行向量检索。

流程：

```text
query
-> embedding_model.embed_query
-> Milvus search top 10
-> 拼接 context
-> rag_chain 生成答案
```

测试问题：

```text
如果我想要一个更可控的多步骤 Agent，应该选什么？
```

当 Milvus 只返回 top 5 时，关键 chunk 没有排在前面，回答偏保守。将 Milvus 召回数量提高到 top 10 后，关键 chunk 出现在上下文中，模型能够正确回答 LangGraph。

这说明：

```text
Milvus naive search 不是不能用，而是 top-k 太小时容易漏掉关键 chunk。
提高 candidate_k 可以改善召回，但也会引入更多上下文噪声。
```

## `/ask_milvus_rerank`

`/ask_milvus_rerank` 结合 Milvus 和 CrossEncoder reranker。

流程：

```text
query
-> Milvus search top 10 candidates
-> CrossEncoder rerank
-> 取 top 3 chunks
-> rag_chain 生成答案
```

测试问题：

```text
如果我想要一个更可控的多步骤 Agent，应该选什么？
```

返回结果能够明确回答：

```text
应该选择 LangGraph。
```

返回字段包括：

- `question`
- `candidate_k`
- `rerank_top_k`
- `answer`
- `retrieved_chunks`
- `milvus_candidate_scores`
- `rerank_scores`

本次测试中，Milvus top 10 提供更广的候选上下文，reranker 将上下文压缩到 top 3。最终答案能正确指出 LangGraph 是更适合可控多步骤 Agent 的选择。

## 关键观察

- Milvus 可以替换 Chroma 的向量检索层，但不需要重写整个 RAG 系统。
- Milvus standalone 依赖 etcd 和 MinIO，部署复杂度明显高于 Chroma。
- Docker 镜像拉取可能是本地环境中的主要阻塞点。
- Milvus search 的 top-k 设置会显著影响最终答案质量。
- 单纯替换向量数据库不一定直接提升 RAG 效果。
- Milvus + Rerank 更符合生产级 RAG 的两阶段检索思路。
- 保留 Chroma baseline 有助于比较不同检索后端的效果。

## 今日结论

Day 12 完成了从 Chroma 到 Milvus 的关键过渡。

本日完成内容：

- 理解 Chroma 和 Milvus 的定位差异
- 启动 Milvus standalone
- 解决 Docker Hub 镜像拉取问题
- 使用 `milvus_demo.py` 跑通 collection 创建、insert 和 search
- 从 Chroma 读取文档并写入 Milvus
- 新增 `/ask_milvus`
- 新增 `/ask_milvus_rerank`

最终结论：

```text
Milvus 可以作为生产级向量数据库替代 Chroma 的检索层。
但生产级向量库不自动等于更好的答案。
RAG 效果仍然取决于 top-k、rerank、chunk 质量和上下文构建策略。
```

对于当前项目，较合理的生产级检索路线是：

```text
Milvus 负责大规模候选召回
Reranker 负责精排和过滤噪声
LLM 基于精排后的上下文生成答案
```

## 下一步

Day 13 计划进入复杂文档处理，重点关注 PDF、Markdown、HTML 等不同格式文档的加载、清洗、切分和 metadata 保留。
