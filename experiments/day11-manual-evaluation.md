# Day 11 Manual RAG Evaluation

## 背景

Day 10 已经构建了固定测试集，并通过 `eval_manual.py` 自动调用不同 RAG 接口，生成 `eval_results.json`。Day 11 在此基础上进行人工评分，用统一维度比较不同 RAG 策略的表现。

## 今日目标

- 对代表性问题进行人工评分
- 比较 `/ask`、`/ask_multi_query`、`/ask_hyde`、`/ask_rerank`
- 总结不同 RAG 策略的优缺点和适用场景

## 评分文件

评分文件：`eval_scores.csv`

## 评分维度

| 维度 | 说明 | 分值 |
|---|---|---|
| `retrieval_relevance` | `retrieved_chunks` 是否与问题相关 | 1-5 |
| `answer_correctness` | `answer` 是否正确 | 1-5 |
| `groundedness` | `answer` 是否基于 `retrieved_chunks` | 1-5 |
| `completeness` | 是否覆盖 `expected_points` | 1-5 |

## 代表性问题

本次先选择 q1、q7、q8 三个问题进行评分：

- q1：概念对比类问题
- q7：决策类问题
- q8：工具理解类问题

## 评分结果

| 接口 | 样本数 | retrieval_relevance | answer_correctness | groundedness | completeness | overall |
|---|---:|---:|---:|---:|---:|---:|
| `/ask` | 3 | 3.67 | 4.33 | 5.00 | 3.33 | 4.08 |
| `/ask_multi_query` | 3 | 4.33 | 5.00 | 5.00 | 5.00 | 4.83 |
| `/ask_hyde` | 3 | 3.67 | 3.67 | 5.00 | 2.67 | 3.75 |
| `/ask_rerank` | 3 | 4.67 | 5.00 | 5.00 | 4.33 | 4.75 |

## 初步观察

- q1 中 `/ask_rerank` 表现最好，第一条 chunk 命中核心内容。
- q7 中 `/ask` 回答偏保守，而 Advanced RAG 方法更容易给出 LangGraph。
- q8 中 `/ask_hyde` 表现较差，说明 HyDE 并不总是提升效果。
- `/ask_multi_query` 在部分问题上更完整，但耗时明显更高。
- `/ask_rerank` 在效果和耗时之间比较平衡。

## 策略适用场景总结

| 策略 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| Naive RAG | 快、简单、成本低 | 对模糊问题和决策类问题较弱 | 关键词明确、问题直接的场景 |
| Multi-Query | 召回范围更广，答案更完整 | 额外调用 LLM，耗时更高，可能引入噪声 | 多角度问题、决策类问题 |
| HyDE | 能增强短 query 的语义表达 | 不稳定，可能召回不到关键 chunk，耗时最高 | 原始 query 太短或缺少关键词时 |
| Rerank | 改善 chunk 排序，耗时适中 | 需要额外 reranker 模型 | 候选 chunks 较多、需要过滤噪声时 |

## 今日结论

Day 11 完成了第一版人工评分。

从 q1、q7、q8 三个代表性问题来看，`/ask_multi_query` 的综合评分最高，说明多查询改写在概念对比和决策类问题上能提升答案完整性。`/ask_rerank` 综合评分接近 Multi-Query，但平均耗时更低，说明 Rerank 在效果和成本之间更平衡。

`/ask_hyde` 在部分问题上表现不稳定，尤其是 q8 中没有召回关键 chunk，导致最终回答“不知道”。这说明 HyDE 适合作为可选策略，而不一定适合作为默认策略。

综合来看，后续可以考虑默认使用 `/ask_rerank`，在问题较复杂或需要多角度理解时使用 `/ask_multi_query`。

## 下一步

Day 12 计划进入生产级向量数据库，尝试引入 Milvus 或理解 Milvus 与 Chroma 的区别，为后续更大规模文档检索做准备。
