# Day 10 RAG Evaluation

## 背景

Day 8 和 Day 9 分别实现了 Multi-Query、HyDE 和 Rerank。为了比较不同 RAG 策略的效果，需要构建一组固定测试问题，并从检索质量和回答质量两个角度进行评估。

## 今日目标

- 构建 RAG 测试集
- 使用同一组问题对比 `/ask`、`/ask_multi_query`、`/ask_hyde`、`/ask_rerank`
- 记录每个接口的成功率、响应耗时和代表性表现
- 为 Day 11 的人工评分提供基础数据

## 评估数据集

测试集文件：`eval_questions.json`

测试集包含 8 个问题，每个问题包含：

- `id`
- `query`
- `expected_points`
- `category`

## 自动调用脚本

本次使用 `eval_manual.py` 自动调用 4 个接口，并记录每次请求的状态码、耗时和返回结果。

输出文件：`eval_results.json`

该文件用于后续人工评分和结果分析。

## 评估维度

| 维度 | 说明 | 分值 |
|---|---|---|
| `retrieval_relevance` | `retrieved_chunks` 是否与问题相关 | 1-5 |
| `answer_correctness` | `answer` 是否正确 | 1-5 |
| `groundedness` | `answer` 是否基于 `retrieved_chunks` | 1-5 |
| `completeness` | 是否覆盖 `expected_points` | 1-5 |
| `latency` | 响应耗时 | 记录秒数 |

## 手工评分规则

- 5 分：表现很好，基本无问题
- 4 分：整体可用，有轻微缺失
- 3 分：部分可用，但有明显不足
- 2 分：相关性较弱或答案明显不完整
- 1 分：基本无效或严重错误

## 自动调用结果汇总

本次共测试 8 个问题，每个问题分别调用 4 个接口。所有接口均成功返回，成功率为 100%。

| 接口 | 测试数量 | 成功数量 | 平均耗时 | 最短耗时 | 最长耗时 |
|---|---:|---:|---:|---:|---:|
| `/ask` | 8 | 8 | 7.036s | 3.292s | 17.730s |
| `/ask_multi_query` | 8 | 8 | 13.289s | 9.047s | 19.988s |
| `/ask_hyde` | 8 | 8 | 16.552s | 8.544s | 33.440s |
| `/ask_rerank` | 8 | 8 | 8.975s | 5.205s | 13.364s |

## 代表性案例分析

### q1：LangChain 和 LangGraph 有什么区别？

四个接口都能回答核心区别。其中 `/ask_rerank` 的第一条 chunk 直接命中 `LangChain vs. LangGraph vs. Deep Agents`，且最高 rerank score 约为 0.986，说明 reranker 对强相关 chunk 的排序效果较明显。

### q7：如果我想要一个更可控的多步骤 Agent，应该选什么？

`/ask` 没有直接给出 LangGraph，回答偏保守。而 `/ask_multi_query`、`/ask_hyde` 和 `/ask_rerank` 都能明确回答 LangGraph。这个案例说明 Advanced RAG 对决策类、意图需要扩展的问题更有帮助。

### q8：MCP 在文档问答中有什么作用？

`/ask_hyde` 返回“不知道”，而其他接口能回答 MCP 可以连接文档到 Claude、VSCode 等工具以获得实时答案。这个案例说明 HyDE 并不稳定，如果 hypothetical document 没有帮助召回关键 chunk，效果可能下降。

## 关键观察

- `/ask` 速度最快，适合作为 baseline。
- `/ask_multi_query` 平均耗时明显增加，因为它需要先调用 LLM 生成多个 rewritten queries。
- `/ask_hyde` 平均耗时最高，因为它需要先生成 hypothetical document，再进行检索和回答。
- `/ask_rerank` 比 `/ask` 慢一些，但整体仍明显快于 Multi-Query 和 HyDE。
- 对于 q7「如果我想要一个更可控的多步骤 Agent，应该选什么？」，`/ask` 没有直接给出 LangGraph，而 `/ask_multi_query`、`/ask_hyde`、`/ask_rerank` 都能更明确地回答 LangGraph，说明 Advanced RAG 对决策类问题有帮助。
- 对于 q8「MCP 在文档问答中有什么作用？」，`/ask_hyde` 返回了“不知道”，说明 HyDE 并不总是提升效果。hypothetical document 如果没有召回到关键 chunk，最终答案可能反而变差。
- Rerank 返回的 `rerank_scores` 可以帮助观察排序置信度。例如 q1 中 `/ask_rerank` 的最高分约为 0.986，说明 reranker 对第一条 chunk 相关性判断较强。

## 今日结论

Day 10 完成了第一版 RAG Evaluation。

通过固定测试集和自动调用脚本，可以对不同 RAG 策略进行更稳定的横向比较。实验结果显示，Naive RAG 速度最快，适合作为 baseline；Multi-Query 和 HyDE 能增强部分问题的召回能力，但会显著增加耗时；Rerank 在耗时增加较小的情况下，可以改善部分问题的 chunk 排序。

本次评估也说明，Advanced RAG 不一定总是优于 Naive RAG，需要根据问题类型、召回结果和成本进行选择。

## 下一步

Day 11 计划在 Day 10 的基础上补充人工评分表，对不同接口的 retrieval relevance、answer correctness、groundedness 和 completeness 进行量化评分，并总结不同 RAG 策略的适用场景。
