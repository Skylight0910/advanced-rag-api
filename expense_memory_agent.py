import json

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver

from expense_db import init_database, sql_query

load_dotenv()
load_dotenv("/app/.env")


@tool
def query_expenses(sql: str) -> str:
    """执行个人消费记录的只读 SQLite 查询。仅支持 SELECT 或 WITH SQL。"""
    return json.dumps(
        sql_query(sql),
        ensure_ascii=False,
    )


model = ChatDeepSeek(
    model="deepseek-v4-pro",
    temperature=0,
)

agent = create_agent(
    model=model,
    tools=[query_expenses],
    system_prompt="""
你是个人消费查询助手。

数据库表结构：
expenses(
    id INTEGER,
    expense_date TEXT,
    category TEXT,
    amount REAL,
    description TEXT,
    payment_method TEXT
)

规则：
1. 查询消费记录时必须调用 query_expenses。
2. 只能生成 SELECT 或 WITH 查询。
3. 禁止 INSERT、UPDATE、DELETE、DROP、ALTER。
4. 必须严格根据查询结果回答。
""",
    checkpointer=InMemorySaver(),
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
    init_database()

    thread_id = "expense-demo-1"

    questions = [
        "餐饮一共花了多少钱？",
        "其中最贵的一笔是什么？",
        "它是用什么支付方式付款的？",
    ]

    for question in questions:
        print("=" * 80)
        print("User:", question)
        print("Agent:", ask(question, thread_id))


if __name__ == "__main__":
    main()
