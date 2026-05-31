# Day 9 Rerank

## 背景

Day 8 实现了 Query Transformation，包括 Multi-Query 和 HyDE，用于增强检索阶段的召回能力。

但在 RAG 系统中，只“召回更多文档”并不一定意味着最终回答更好。如果召回结果中混入了不相关或弱相关的 chunk，模型可能会受到噪声影响。因此 Day 9 引入 Rerank：先召回更多候选文档，再使用 reranker 对候选文档重新排序，选择最相关的文档块作为最终上下文。

本次实验采用的思路是：

```text
Retriever 负责召回候选文档
Reranker 负责重新排序候选文档
LLM 基于 rerank 后的 top chunks 生成答案
今日目标
在现有 RAG API 中新增 /ask_rerank 接口。

目标：

使用 Chroma 先召回更多候选文档
使用 CrossEncoder reranker 对候选文档重新打分
取 rerank 后分数最高的 top chunks 作为上下文
对比 /ask 和 /ask_rerank 的 retrieved_chunks 与 answer 差异
改动内容
新增依赖
在 app.py 中引入：

from sentence_transformers import CrossEncoder
新增配置
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
RETRIEVER_CANDIDATE_K = 10
RERANK_TOP_K = 3
含义：

RETRIEVER_CANDIDATE_K = 10：先从向量库中召回 10 个候选文档块
RERANK_TOP_K = 3：rerank 后取分数最高的 3 个文档块作为最终上下文
新增全局 Reranker
reranker = CrossEncoder(RERANKER_MODEL_NAME)
新增 rerank 函数
def rerank_docs(query: str, docs, top_k: int = RERANK_TOP_K):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = sorted(
        zip(docs, scores),
        key=lambda x: float(x[1]),
        reverse=True
    )

    return scored_docs[:top_k]
处理流程：

将用户 query 与每个候选 doc 组成 query-document pair
使用 CrossEncoder 给每个 pair 打相关性分数
按分数从高到低排序
返回 top-k 个最相关文档块
新增接口：POST /ask_rerank
处理流程：

接收用户输入 query
使用 Chroma similarity_search 召回 10 个候选文档块
使用 CrossEncoder reranker 对候选文档重新排序
取 rerank 后 top 3 文档块
将文档块拼接成 context
使用现有 rag_chain 生成最终答案
返回答案、最终 chunks 和 rerank 分数
返回字段：

question
answer
retrieved_chunks
rerank_scores
接口测试
Naive RAG
curl http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"LangChain 和 LangGraph 有什么区别？"}'
Rerank RAG
curl http://localhost:8000/ask_rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"LangChain 和 LangGraph 有什么区别？"}'
测试问题
本次使用以下问题进行对比：

LangChain 和 LangGraph 有什么区别？
Deep Agents 适合什么场景？
什么情况下应该使用 LangGraph？
这个知识库主要讲了什么？
对比维度
本次实验主要从以下维度观察效果：

/ask 和 /ask_rerank 返回的 retrieved_chunks 是否不同
rerank 后的 chunks 是否更贴近用户问题
rerank_scores 是否能体现相关性排序
最终 answer 是否更准确或更完整
是否减少了无关上下文
是否增加了响应耗时
模型启动时间是否增加
实验结果
测试问题	方法	检索策略	retrieved_chunks 相关性	回答质量	观察
LangChain 和 LangGraph 有什么区别？	/ask	Chroma top 5	可用	可用	直接使用向量相似度排序
LangChain 和 LangGraph 有什么区别？	/ask_rerank	Chroma top 10 + rerank top 3	较好	较完整	rerank 后更相关的 chunk 排在前面
Deep Agents 适合什么场景？	/ask	Chroma top 5	可用	可用	原始 query 明确时效果已可接受
Deep Agents 适合什么场景？	/ask_rerank	Chroma top 10 + rerank top 3	较好	较完整	rerank 有助于过滤弱相关 chunk
什么情况下应该使用 LangGraph？	/ask	Chroma top 5	一般	可用	可能召回到泛化的 LangChain 内容
什么情况下应该使用 LangGraph？	/ask_rerank	Chroma top 10 + rerank top 3	较好	更聚焦	更容易保留和 LangGraph 直接相关的片段
关键观察
Rerank 的核心作用不是“召回更多”，而是“重新排序”。
Chroma similarity_search 适合作为第一阶段召回，速度较快。
CrossEncoder reranker 会同时读取 query 和 document，因此比单纯向量相似度更适合判断细粒度相关性。
/ask_rerank 使用 top 10 候选，再取 top 3，可以在召回范围和上下文噪声之间取得平衡。
rerank 后的 retrieved_chunks 更适合用来构建最终 prompt。
Reranker 会增加模型加载时间和推理成本，因此不一定适合所有场景。
如果原始 query 已经非常明确，Naive RAG 和 Rerank RAG 的差异可能不明显。
如果原始检索结果包含多个弱相关 chunk，Rerank 更容易体现价值。
遇到的问题
BAAI/bge-reranker-base 首次加载需要从 HuggingFace 下载模型，可能受到网络或代理影响。
Reranker 模型会增加服务启动时间。
本地调试时如果使用 uvicorn --reload，模型可能被重复加载，建议手动重启服务。
Rerank 阶段需要处理 query-doc pair，因此候选文档数量越多，推理成本越高。
当前实现只返回最终 top chunks，没有返回 rerank 前的候选 chunks，后续可以增加调试字段用于更细致对比。
今日结论
Day 9 完成了 Rerank RAG 的基础实现。

通过新增 /ask_rerank 接口，系统从原来的单阶段向量检索升级为两阶段检索：

Chroma 召回候选文档 -> CrossEncoder 重新排序 -> LLM 基于 top chunks 回答
Rerank 的价值在于提高最终上下文的相关性，减少弱相关 chunk 对回答的干扰。它尤其适合候选文档较多、问题较复杂、向量相似度排序不够精确的场景。

同时，Rerank 会带来额外的模型加载时间和推理成本，因此需要根据实际场景决定是否启用。

下一步
Day 10 计划进入 RAG Evaluation，构建一组固定测试问题和评价维度，用于系统性比较不同 RAG 策略的回答质量与检索效果。
