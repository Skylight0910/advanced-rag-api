import json

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek

from expense_db import init_database, sql_query

load_dotenv()
load_dotenv("/app/.env")

MAX_STEPS = 5


@tool
def query_expenses(sql: str) -> str:
    """执行个人消费记录的只读 SQLite 查询。仅支持 SELECT 或 WITH SQL。"""
    return json.dumps(
        sql_query(sql),
        ensure_ascii=False,
    )


TOOLS = [query_expenses]
TOOL_MAP = {item.name: item for item in TOOLS}

llm = ChatDeepSeek(
    model="deepseek-v4-pro",
    temperature=0,
)

llm_with_tools = llm.bind_tools(TOOLS)


def run_agent(user_input: str) -> str:
    messages = [
        SystemMessage(
            content="""
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
3. 不允许生成 INSERT、UPDATE、DELETE、DROP、ALTER。
4. 最多返回 20 行。
5. 必须严格根据工具结果回答。
"""
        ),
        HumanMessage(content=user_input),
    ]

    for step in range(MAX_STEPS):
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        print(f"\nStep {step + 1}:")
        print("Tool calls:")
        print(json.dumps(ai_message.tool_calls, ensure_ascii=False, indent=2))

        if not ai_message.tool_calls:
            return str(ai_message.content)

        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]

            if tool_name not in TOOL_MAP:
                raise ValueError(f"Unknown tool: {tool_name}")

            tool_message = TOOL_MAP[tool_name].invoke(tool_call)
            messages.append(tool_message)

            print("\nTool result:")
            print(tool_message.content)

    raise RuntimeError("Agent exceeded maximum steps")


def main():
    init_database()

    questions = [
        "餐饮一共花了多少钱？",
        "按类别统计支出，并从高到低排序。",
        "有哪些超过 100 元的消费？",
    ]

    for question in questions:
        print("=" * 80)
        print("User:", question)
        print("\nFinal answer:")
        print(run_agent(question))
        print()


if __name__ == "__main__":
    main()
