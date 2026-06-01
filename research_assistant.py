import json
import os

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
load_dotenv("/app/.env")

RAG_ENDPOINT = "http://localhost:8000/ask_hybrid_rerank"


@tool
def rag_search(query: str) -> str:
    """查询个人知识库。适合检索项目笔记、故障记录和已导入文档。"""
    response = requests.post(
        RAG_ENDPOINT,
        json={"query": query},
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()

    return json.dumps(
        {
            "answer": data.get("answer", ""),
            "retrieved_chunks": data.get("retrieved_chunks", [])[:3],
        },
        ensure_ascii=False,
    )


@tool
def web_search(query: str) -> str:
    """搜索互联网最新信息。适合查询知识库之外的信息、实时信息和外部资料。"""
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("缺少 TAVILY_API_KEY")

    response = requests.post(
        "https://api.tavily.com/search",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    return json.dumps(
        {
            "query": data.get("query"),
            "results": [
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": item.get("content"),
                }
                for item in data.get("results", [])
            ],
        },
        ensure_ascii=False,
    )


model = ChatDeepSeek(
    model="deepseek-v4-pro",
    temperature=0,
)

agent = create_agent(
    model=model,
    tools=[rag_search, web_search],
    system_prompt="""
你是研究助手。

工具选择规则：
1. 查询个人项目、学习笔记和故障记录时，使用 rag_search。
2. 查询外部资料或实时信息时，使用 web_search。
3. 用户要求结合个人资料和外部信息时，可以依次调用两个工具。
4. 必须根据工具结果回答，不要编造信息。
5. 使用 Web 搜索结果时，保留来源 URL。
""",
    checkpointer=InMemorySaver(),
    middleware=[
        ModelCallLimitMiddleware(
            run_limit=6,
            exit_behavior="end",
        ),
        ToolCallLimitMiddleware(
            run_limit=5,
            exit_behavior="continue",
        ),
        ToolCallLimitMiddleware(
            tool_name="web_search",
            run_limit=2,
            exit_behavior="continue",
        ),
        ToolRetryMiddleware(
            tools=["rag_search", "web_search"],
            max_retries=2,
            initial_delay=1.0,
            backoff_factor=2.0,
        ),
    ],
)


def ask(question: str, thread_id: str) -> str:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id,
            }
        },
    )

    return str(result["messages"][-1].content)


def main():
    thread_id = "research-demo-1"

    questions = [
        "根据我的知识库，Milvus 插入成功但搜索不到结果应该怎么办？",
        "搜索互联网，介绍一下 Tavily Search API 的用途，并附上来源链接。",
        "结合我的知识库和互联网资料，解释 RAG 与 Web 搜索工具有什么区别。",
    ]

    for question in questions:
        print("=" * 80)
        print("User:", question)
        print("Agent:", ask(question, thread_id))
        print()


if __name__ == "__main__":
    main()
