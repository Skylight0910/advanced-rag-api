import os
import jieba
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from pymilvus import MilvusClient
from sentence_transformers import CrossEncoder

from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
load_dotenv("/app/.env")

# ------------------ 配置 ------------------
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
LLM_MODEL = "deepseek-v4-pro"

RETRIEVER_K = 5
RETRIEVER_CANDIDATE_K = 10
RERANK_TOP_K = 3

HYBRID_BM25_K = 10
HYBRID_EMBEDDING_K = 10
HYBRID_RRF_K = 60

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"
MILVUS_COLLECTION_NAME = "rag_chunks_collection"
REBUILD_MILVUS_ON_START = False

# ------------------ 启动时加载全局资源 ------------------
print("⏳ 正在加载 Embedding 模型...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

print("⏳ 正在加载 Chroma 向量数据库...")
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
)

print("⏳ 正在连接 Milvus...")
milvus_client = MilvusClient(
    uri=MILVUS_URI,
    token=MILVUS_TOKEN,
)

print("⏳ 正在初始化 DeepSeek LLM...")
llm = ChatDeepSeek(
    model=LLM_MODEL,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.3,
)

print("⏳ 正在加载 Reranker 模型...")
reranker = CrossEncoder(RERANKER_MODEL_NAME)

# ------------------ RAG Chains ------------------
rag_prompt = ChatPromptTemplate.from_template("""你是一个细心的文档助手。请根据以下上下文回答用户的问题。
如果上下文中没有答案，请如实说不知道，不要编造。

上下文：
{context}

问题：{question}
回答：""")

rag_chain = rag_prompt | llm | StrOutputParser()

multi_query_prompt = ChatPromptTemplate.from_template("""请根据用户问题，生成 3 个语义相关但角度不同的检索查询。
要求：
- 每行只输出一个查询
- 不要输出编号
- 不要输出解释

用户问题：{question}
检索查询：""")

multi_query_chain = multi_query_prompt | llm | StrOutputParser()

hyde_prompt = ChatPromptTemplate.from_template("""请根据用户问题，写一段可能出现在知识库中的简短说明。
这段内容只用于向量检索，不是最终答案。

用户问题：{question}
假想文档：""")

hyde_chain = hyde_prompt | llm | StrOutputParser()


# ------------------ 工具函数 ------------------
class TextDoc:
    def __init__(self, page_content: str):
        self.page_content = page_content

bm25_index = None
bm25_docs = []
def tokenize_for_bm25(text: str) -> list[str]:
    return [
        token.strip().lower()
        for token in jieba.lcut(text)
        if token.strip()
    ]


def init_bm25_from_chroma():
    global bm25_index, bm25_docs

    chroma_data = vectordb.get(include=["documents"])
    texts = [
        text
        for text in chroma_data.get("documents", [])
        if text
    ]

    bm25_docs = [TextDoc(text) for text in texts]
    tokenized_corpus = [
        tokenize_for_bm25(text)
        for text in texts
    ]

    if not tokenized_corpus:
        print("⚠️ Chroma 中没有文档，BM25 索引为空")
        return

    bm25_index = BM25Okapi(tokenized_corpus)
    print(f"✅ BM25 索引初始化完成，共 {len(bm25_docs)} 个 chunks")


def bm25_search(query: str, k: int = HYBRID_BM25_K):
    if bm25_index is None:
        return []

    scores = bm25_index.get_scores(tokenize_for_bm25(query))
    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )[:k]

    return [
        (bm25_docs[index], float(scores[index]))
        for index in ranked_indexes
        if float(scores[index]) > 0
    ]


def hybrid_search(
    query: str,
    bm25_k: int = HYBRID_BM25_K,
    embedding_k: int = HYBRID_EMBEDDING_K,
):
    embedding_docs = vectordb.similarity_search(query, k=embedding_k)
    keyword_results = bm25_search(query, k=bm25_k)

    docs_by_text = {}
    rrf_scores = {}
    sources = {}

    def add_result(doc, rank: int, source: str):
        text = doc.page_content
        docs_by_text[text] = doc
        rrf_scores[text] = rrf_scores.get(text, 0.0) + (
            1.0 / (HYBRID_RRF_K + rank)
        )
        sources.setdefault(text, set()).add(source)

    for rank, doc in enumerate(embedding_docs, start=1):
        add_result(doc, rank, "embedding")

    for rank, (doc, _) in enumerate(keyword_results, start=1):
        add_result(doc, rank, "bm25")

    ranked_texts = sorted(
        docs_by_text,
        key=lambda text: rrf_scores[text],
        reverse=True,
    )

    return [
        {
            "doc": docs_by_text[text],
            "rrf_score": rrf_scores[text],
            "sources": sorted(sources[text]),
        }
        for text in ranked_texts
    ]
        
def parse_rewritten_queries(text: str) -> list[str]:
    queries = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        line = line.lstrip("-*0123456789.、) ")
        if line:
            queries.append(line)

    return queries[:3]


def dedupe_docs(docs):
    seen = set()
    unique_docs = []

    for doc in docs:
        if doc.page_content in seen:
            continue

        seen.add(doc.page_content)
        unique_docs.append(doc)

    return unique_docs


def build_context(docs) -> str:
    return "\n\n".join([doc.page_content for doc in docs])


def build_context_from_texts(texts: list[str]) -> str:
    return "\n\n".join(texts)


def preview_chunks(docs) -> list[str]:
    return [doc.page_content[:200] for doc in docs]


def preview_texts(texts: list[str]) -> list[str]:
    return [text[:200] for text in texts]


def rerank_docs(query: str, docs, top_k: int = RERANK_TOP_K):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = sorted(
        zip(docs, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    return scored_docs[:top_k]


def init_milvus_from_chroma():
    sample_vector = embedding_model.embed_query("测试")
    dim = len(sample_vector)

    if milvus_client.has_collection(MILVUS_COLLECTION_NAME):
        if REBUILD_MILVUS_ON_START:
            print("⏳ 正在删除已有 Milvus collection...")
            milvus_client.drop_collection(MILVUS_COLLECTION_NAME)
        else:
            milvus_client.load_collection(MILVUS_COLLECTION_NAME)
            print("✅ Milvus collection 已存在，跳过重建")
            return

    print("⏳ 正在创建 Milvus collection...")
    milvus_client.create_collection(
        collection_name=MILVUS_COLLECTION_NAME,
        dimension=dim,
    )

    print("⏳ 正在从 Chroma 读取文档并写入 Milvus...")
    chroma_data = vectordb.get(include=["documents"])
    documents = chroma_data.get("documents", [])

    data = []
    for i, text in enumerate(documents):
        if not text:
            continue

        data.append({
            "id": i,
            "vector": embedding_model.embed_query(text),
            "text": text,
        })

    if data:
        insert_result = milvus_client.insert(
            collection_name=MILVUS_COLLECTION_NAME,
            data=data,
        )
        print("Milvus insert result:", insert_result)
    else:
        print("⚠️ 没有从 Chroma 读取到文档，Milvus collection 为空")

    milvus_client.flush(collection_name=MILVUS_COLLECTION_NAME)
    milvus_client.load_collection(collection_name=MILVUS_COLLECTION_NAME)
    print("✅ Milvus collection 初始化完成")


def milvus_search(query: str, k: int = RETRIEVER_K):
    query_vector = embedding_model.embed_query(query)

    results = milvus_client.search(
        collection_name=MILVUS_COLLECTION_NAME,
        data=[query_vector],
        limit=k,
        output_fields=["text"],
    )

    hits = results[0] if results else []
    texts = [hit["entity"]["text"] for hit in hits]
    scores = [float(hit["distance"]) for hit in hits]

    return texts, scores

def milvus_rerank_search(
    query: str,
    candidate_k: int = RETRIEVER_CANDIDATE_K,
    top_k: int = RERANK_TOP_K,
):
    texts, milvus_scores = milvus_search(query, k=candidate_k)
    docs = [TextDoc(text) for text in texts]

    reranked_docs = rerank_docs(query, docs, top_k=top_k)
    final_docs = [doc for doc, score in reranked_docs]

    return {
        "texts": [doc.page_content for doc in final_docs],
        "rerank_scores": [float(score) for doc, score in reranked_docs],
        "milvus_candidate_scores": milvus_scores,
    }
    
init_milvus_from_chroma()
init_bm25_from_chroma()

print("✅ 所有组件加载完毕，API 就绪！")

# ------------------ FastAPI 应用 ------------------
app = FastAPI(title="Advanced RAG API", version="2.0.0")


class Question(BaseModel):
    query: str


@app.post("/ask")
async def ask_question(q: Question):
    docs = vectordb.similarity_search(q.query, k=RETRIEVER_K)
    context = build_context(docs)

    answer = rag_chain.invoke({
        "context": context,
        "question": q.query,
    })

    return {
        "question": q.query,
        "answer": answer,
        "retrieved_chunks": preview_chunks(docs),
    }


@app.post("/ask_multi_query")
async def ask_multi_query(q: Question):
    rewritten_text = multi_query_chain.invoke({"question": q.query})
    rewritten_queries = parse_rewritten_queries(rewritten_text)

    if not rewritten_queries:
        rewritten_queries = [q.query]

    all_docs = []
    for query in rewritten_queries:
        docs = vectordb.similarity_search(query, k=RETRIEVER_K)
        all_docs.extend(docs)

    unique_docs = dedupe_docs(all_docs)
    context = build_context(unique_docs)

    answer = rag_chain.invoke({
        "context": context,
        "question": q.query,
    })

    return {
        "question": q.query,
        "rewritten_queries": rewritten_queries,
        "answer": answer,
        "retrieved_chunks": preview_chunks(unique_docs),
    }


@app.post("/ask_hyde")
async def ask_hyde(q: Question):
    hypothetical_doc = hyde_chain.invoke({"question": q.query})

    docs = vectordb.similarity_search(hypothetical_doc, k=RETRIEVER_K)
    context = build_context(docs)

    answer = rag_chain.invoke({
        "context": context,
        "question": q.query,
    })

    return {
        "question": q.query,
        "hypothetical_doc": hypothetical_doc,
        "answer": answer,
        "retrieved_chunks": preview_chunks(docs),
    }


@app.post("/ask_rerank")
async def ask_rerank(q: Question):
    candidate_docs = vectordb.similarity_search(
        q.query,
        k=RETRIEVER_CANDIDATE_K,
    )

    reranked_docs = rerank_docs(q.query, candidate_docs)
    final_docs = [doc for doc, score in reranked_docs]
    context = build_context(final_docs)

    answer = rag_chain.invoke({
        "context": context,
        "question": q.query,
    })

    return {
        "question": q.query,
        "candidate_k": RETRIEVER_CANDIDATE_K,
        "rerank_top_k": RERANK_TOP_K,
        "answer": answer,
        "retrieved_chunks": preview_chunks(final_docs),
        "rerank_scores": [float(score) for doc, score in reranked_docs],
    }


@app.post("/ask_milvus")
async def ask_milvus(q: Question):
    texts, scores = milvus_search(q.query, k=10)
    context = build_context_from_texts(texts)

    answer = rag_chain.invoke({
        "context": context,
        "question": q.query,
    })

    return {
        "question": q.query,
        "answer": answer,
        "retrieved_chunks": preview_texts(texts),
        "milvus_scores": scores,
    }

@app.post("/ask_milvus_rerank")
async def ask_milvus_rerank(q: Question):
    result = milvus_rerank_search(
        q.query,
        candidate_k=RETRIEVER_CANDIDATE_K,
        top_k=RERANK_TOP_K,
    )

    texts = result["texts"]
    context = build_context_from_texts(texts)

    answer = rag_chain.invoke({
        "context": context,
        "question": q.query,
    })

    return {
        "question": q.query,
        "candidate_k": RETRIEVER_CANDIDATE_K,
        "rerank_top_k": RERANK_TOP_K,
        "answer": answer,
        "retrieved_chunks": preview_texts(texts),
        "milvus_candidate_scores": result["milvus_candidate_scores"],
        "rerank_scores": result["rerank_scores"],
    }

@app.post("/ask_hybrid")
async def ask_hybrid(q: Question):
    results = hybrid_search(q.query)
    final_results = results[:RETRIEVER_K]
    docs = [item["doc"] for item in final_results]

    answer = rag_chain.invoke({
        "context": build_context(docs),
        "question": q.query,
    })

    return {
        "question": q.query,
        "answer": answer,
        "retrieved_chunks": preview_chunks(docs),
        "retrieval_details": [
            {
                "rrf_score": item["rrf_score"],
                "sources": item["sources"],
            }
            for item in final_results
        ],
    }


@app.post("/ask_hybrid_rerank")
async def ask_hybrid_rerank(q: Question):
    hybrid_results = hybrid_search(q.query)
    candidate_docs = [
        item["doc"]
        for item in hybrid_results[:RETRIEVER_CANDIDATE_K]
    ]

    reranked_docs = rerank_docs(q.query, candidate_docs)
    final_docs = [doc for doc, _ in reranked_docs]

    answer = rag_chain.invoke({
        "context": build_context(final_docs),
        "question": q.query,
    })

    return {
        "question": q.query,
        "candidate_k": len(candidate_docs),
        "rerank_top_k": RERANK_TOP_K,
        "answer": answer,
        "retrieved_chunks": preview_chunks(final_docs),
        "rerank_scores": [
            float(score)
            for _, score in reranked_docs
        ],
    }
    
@app.get("/")
def root():
    return {
        "message": "Advanced RAG API is running.",
        "endpoints": [
            "/ask",
            "/ask_multi_query",
            "/ask_hyde",
            "/ask_rerank",
            "/ask_milvus",
            "/ask_milvus_rerank",   
	    "/ask_hybrid",
	    "/ask_hybrid_rerank",
        ],
    }
