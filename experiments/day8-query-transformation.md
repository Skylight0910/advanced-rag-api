# Day 8 Query Transformation

## 背景

第一周实现的 Naive RAG 直接使用用户原始 query 做向量检索。这个方式简单、速度快，但当用户问题较短、表达模糊或缺少关键词时，可能无法召回最相关的文档块。

因此 Day 8 引入 Query Transformation，对用户 query 进行改写或语义扩展，再用于向量检索。本次实验主要实现两种方法：

- Multi-Query：将一个问题改写成多个不同角度的检索 query
- HyDE：先生成一段 hypothetical document，再用它进行向量检索

## 今日目标

在原有 `/ask` Naive RAG 接口基础上，新增两个 Advanced RAG 接口：

- `/ask_multi_query`
- `/ask_hyde`

并通过相同测试问题，对比三种 RAG 方法的检索结果和回答质量。

## 改动内容

### 保留原接口

- `POST /ask`

该接口继续使用原始 query 直接进行向量检索。

### 新增接口：POST /ask_multi_query

处理流程：

1. 接收用户输入 `query`
2. 调用 LLM，将原始问题改写成 3 个语义相关但角度不同的检索 query
3. 分别用 3 个 query 调用 Chroma `similarity_search`
4. 合并所有检索结果
5. 按 `page_content` 去重
6. 将去重后的文档块拼接为 context
7. 使用原始 query 和 context 调用 `rag_chain` 生成最终答案

返回字段：

- `question`
- `rewritten_queries`
- `answer`
- `retrieved_chunks`

### 新增接口：POST /ask_hyde

处理流程：

1. 接收用户输入 `query`
2. 调用 LLM，根据 query 生成一段 hypothetical document
3. 该 hypothetical document 只用于检索，不作为最终答案
4. 使用 hypothetical document 调用 Chroma `similarity_search`
5. 将检索到的文档块拼接为 context
6. 使用原始 query 和 context 调用 `rag_chain` 生成最终答案

返回字段：

- `question`
- `hypothetical_doc`
- `answer`
- `retrieved_chunks`

## 接口测试

### Naive RAG

```bash
curl http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"LangChain 和 LangGraph 有什么区别？"}'
Multi-Query RAG
curl http://localhost:8000/ask_multi_query \
  -H "Content-Type: application/json" \
  -d '{"query":"LangChain 和 LangGraph 有什么区别？"}'
HyDE RAG
curl http://localhost:8000/ask_hyde \
  -H "Content-Type: application/json" \
  -d '{"query":"LangChain 和 LangGraph 有什么区别？"}'
测试问题
本次使用以下问题进行对比：

LangChain 和 LangGraph 有什么区别？
Deep Agents 适合什么场景？
这个知识库主要讲了什么？
对比维度
本次实验主要从以下维度观察效果：

retrieved_chunks 是否与问题相关
answer 是否覆盖问题核心点
回答是否基于检索到的上下文
是否出现无依据内容
相比 Naive RAG，回答是否更完整
是否引入更多噪声
是否增加额外 LLM 调用成本
实验结果
测试问题	方法	检索效果	回答质量	观察
LangChain 和 LangGraph 有什么区别？	/ask	可用	可用	直接检索，结果较简洁
LangChain 和 LangGraph 有什么区别？	/ask_multi_query	较好	更完整	多个 query 提高了召回范围
LangChain 和 LangGraph 有什么区别？	/ask_hyde	较好	较自然	hypothetical document 增强了语义表达
Deep Agents 适合什么场景？	/ask	可用	可用	如果原问题关键词明确，Naive RAG 已经足够
Deep Agents 适合什么场景？	/ask_multi_query	较好	较完整	能补充不同角度的相关内容
Deep Agents 适合什么场景？	/ask_hyde	可用	可用	效果依赖 hypothetical document 的生成质量
关键观察
Naive RAG 简单直接，适合问题本身关键词明确的情况。
Multi-Query 能从多个角度扩展用户问题，召回范围更广。
Multi-Query 可能会引入更多文档块，因此需要注意噪声问题。
HyDE 适合原始 query 较短、语义不够充分的场景。
HyDE 的第一步生成内容不是最终答案，只是用于增强检索。
retrieved_chunks 对调试 RAG 非常重要，可以用来判断答案是否有依据。
Advanced RAG 通常会增加额外 LLM 调用，因此需要在效果、延迟和成本之间权衡。
遇到的问题
本地运行时需要确保 .env 被正确加载，否则 DeepSeek API key 读取失败。
使用 uvicorn --reload 时，Embedding 模型可能被重复加载，调试时手动重启服务更稳定。
Linux 虚拟机配置代理后，localhost 请求可能被代理转发，需要设置 NO_PROXY。
Docker 中的代码不会因为宿主机 app.py 修改而自动更新，除非重新 build 或使用 volume 挂载。
今日结论
Day 8 完成了 Query Transformation 的基础实践。

通过 /ask、/ask_multi_query 和 /ask_hyde 的对比，可以看到 Query Transformation 的核心价值是增强检索阶段的召回能力。Multi-Query 更适合需要多角度理解的问题，HyDE 更适合短问题或语义不充分的问题。

但 Advanced RAG 并不是一定优于 Naive RAG。它会带来额外的 LLM 调用、延迟和成本，也可能引入噪声。因此实际使用时需要根据问题类型和业务场景选择合适策略。

下一步
Day 9 计划继续优化检索结果排序，引入 Rerank 或 Hybrid Search，对比不同检索策略下 retrieved_chunks 的相关性和最终回答质量。
