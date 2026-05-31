# Day 9 补充实验：Hybrid Search

## 实验目标

补齐 Advanced RAG API 的混合检索能力：

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
Top 3 chunks -> LLM
```

## 新增接口

| 接口 | 流程 |
|---|---|
| `/ask_hybrid` | BM25 + Embedding -> RRF -> Top 5 -> LLM |
| `/ask_hybrid_rerank` | BM25 + Embedding -> RRF -> Top 10 -> CrossEncoder -> Top 3 -> LLM |

## 技术决策

### 为什么同时使用 BM25 和 Embedding

- BM25 擅长处理精确关键词、专有名词和版本号。
- Embedding 擅长处理同义表达和语义相似问题。
- 两路召回互补，可以降低仅依赖单一检索方式造成的漏召回风险。

### 为什么使用 jieba

中文文本通常不使用空格分词。BM25 需要 token 序列，因此使用 `jieba` 对中文 query 和文档 chunks 分词。

### 为什么使用 RRF

BM25 分数和向量相似度不是同一种量纲，不适合直接相加。RRF（Reciprocal Rank Fusion）只依赖候选文档在每路结果中的排名：

```text
RRF score = sum(1 / (k + rank))
```

本项目使用 `k = 60`。

## 延迟测试

固定测试集包含 8 个问题，每个接口调用 8 次。

| 接口 | 样本数 | 平均耗时 |
|---|---:|---:|
| `/ask` | 8 | 5.258s |
| `/ask_multi_query` | 8 | 9.855s |
| `/ask_hyde` | 8 | 11.687s |
| `/ask_rerank` | 8 | 8.203s |
| `/ask_hybrid` | 8 | 4.355s |
| `/ask_hybrid_rerank` | 8 | 6.960s |

## 结果分析

- `/ask_multi_query` 和 `/ask_hyde` 需要额外调用 LLM，因此平均延迟较高。
- `/ask_hybrid` 只增加本地 BM25 检索和 RRF 融合，本轮测试中延迟较低。
- `/ask_hybrid_rerank` 增加 CrossEncoder 精排，延迟高于 `/ask_hybrid`，但候选排序更加可靠。
- `/ask_hybrid` 本轮比 `/ask` 更快不代表其天然更快。两者都需要一次 LLM 生成，DeepSeek 接口响应波动会影响小样本平均值。
- 对宽泛问题，Reranker 分数可能整体较低。精排只能在已有候选中重新排序，无法补回未被召回的内容。

## 面试表达

> 为兼顾精确关键词与语义召回，我实现了 BM25 和 Embedding 的双路检索，并使用 RRF 基于排名进行融合，避免直接混合不同量纲的原始分数。融合后的候选可以继续交给 CrossEncoder 精排，在召回率、排序质量和延迟之间做权衡。

## 完成状态

Hybrid Search 补充实验已完成。后续将在问题驱动练习中构造纯向量检索失败案例，进一步验证混合检索的价值。
