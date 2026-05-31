import ast
import json
import operator
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
load_dotenv("/app/.env")

RAG_ENDPOINT = "http://localhost:8000/ask_rerank"

llm = ChatDeepSeek(
    model="deepseek-v4-pro",
    temperature=0,
)


TOOLS = [
    {
        "name": "rag_search",
        "description": "查询当前 Advanced RAG 项目知识库，适合回答 LangChain、LangGraph、Deep Agents、MCP、RAG、Agent 相关问题。",
        "args": {
            "query": "string，用户要查询的问题"
        },
    },
    {
        "name": "calculator",
        "description": "计算简单数学表达式，例如 12*8+5。",
        "args": {
            "expression": "string，数学表达式"
        },
    },
    {
        "name": "project_status",
        "description": "查看当前项目已经实现了哪些能力和接口。",
        "args": {},
    },
]


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def rag_search(query: str) -> str:
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
            "retrieved_chunks": data.get("retrieved_chunks", [])[:2],
        },
        ensure_ascii=False,
    )


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        value = _safe_eval(node.operand)
        return _ALLOWED_OPERATORS[type(node.op)](value)

    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree.body)
    return str(result)


def project_status() -> str:
    return """当前项目已经支持：
- /ask: Naive RAG
- /ask_multi_query: Multi-Query RAG
- /ask_hyde: HyDE RAG
- /ask_rerank: Chroma + Rerank
- /ask_milvus: Milvus RAG
- /ask_milvus_rerank: Milvus + Rerank
- ingest_docs.py: Markdown/TXT 文档导入
- eval_manual.py / eval_scores.csv: RAG 评估
"""


TOOL_FUNCTIONS = {
    "rag_search": rag_search,
    "calculator": calculator,
    "project_status": project_status,
}


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM output: {text}")

    return json.loads(text[start:end + 1])


def choose_tool(user_input: str) -> dict[str, Any]:
    system_prompt = f"""你是一个工具调用 Agent。你不能直接回答问题，必须先判断是否需要调用工具。

可用工具如下：
{json.dumps(TOOLS, ensure_ascii=False, indent=2)}

请只输出 JSON，不要输出任何解释。

如果需要调用工具，输出：
{{
  "action": "tool_call",
  "tool": "工具名",
  "arguments": {{
    "参数名": "参数值"
  }}
}}

如果不需要工具，输出：
{{
  "action": "final_answer",
  "answer": "你的回答"
}}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ])

    return extract_json(response.content)


def execute_tool(tool_call: dict[str, Any]) -> str:
    tool_name = tool_call.get("tool")
    arguments = tool_call.get("arguments", {})

    if tool_name not in TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool_func = TOOL_FUNCTIONS[tool_name]
    return tool_func(**arguments)


def final_answer(user_input: str, tool_call: dict[str, Any], observation: str) -> str:
    system_prompt = """你是一个 Agent。现在你已经拿到了工具执行结果。
请基于用户问题和工具结果，给出自然语言最终回答。
如果工具结果不足以回答，请说明不足。"""

    content = f"""用户问题：
{user_input}

工具调用：
{json.dumps(tool_call, ensure_ascii=False, indent=2)}

工具结果 observation：
{observation}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=content),
    ])

    return response.content


def run_agent(user_input: str) -> dict[str, Any]:
    tool_call = choose_tool(user_input)

    if tool_call.get("action") == "final_answer":
        return {
            "user_input": user_input,
            "tool_call": tool_call,
            "observation": None,
            "final_answer": tool_call.get("answer", ""),
        }

    observation = execute_tool(tool_call)
    answer = final_answer(user_input, tool_call, observation)

    return {
        "user_input": user_input,
        "tool_call": tool_call,
        "observation": observation,
        "final_answer": answer,
    }


def main():
    questions = [
        "如果我要构建一个更可控的多步骤 Agent，应该用什么？",
        "帮我算一下 12*8+5。",
        "当前项目已经支持哪些能力？",
    ]

    for question in questions:
        print("=" * 80)
        print("User:", question)

        result = run_agent(question)

        print("\nTool call:")
        print(json.dumps(result["tool_call"], ensure_ascii=False, indent=2))

        print("\nObservation:")
        print(result["observation"])

        print("\nFinal answer:")
        print(result["final_answer"])
        print()


if __name__ == "__main__":
    main()
