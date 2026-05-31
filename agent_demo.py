import ast
import operator
import re
from typing import Any

import requests

RAG_ENDPOINT = "http://localhost:8000/ask_rerank"


def rag_search(query: str) -> str:
    response = requests.post(
        RAG_ENDPOINT,
        json={"query": query},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    answer = data.get("answer", "")
    chunks = data.get("retrieved_chunks", [])

    return (
        f"RAG answer:\n{answer}\n\n"
        f"Retrieved chunks preview:\n"
        + "\n---\n".join(chunks[:2])
    )


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
- Naive RAG: /ask
- Multi-Query RAG: /ask_multi_query
- HyDE RAG: /ask_hyde
- Rerank RAG: /ask_rerank
- Milvus RAG: /ask_milvus
- Milvus + Rerank: /ask_milvus_rerank
- RAG Evaluation: eval_questions.json / eval_manual.py / eval_scores.csv
- Document Ingest: ingest_docs.py
"""


def extract_math_expression(text: str) -> str | None:
    match = re.search(r"(\d+(?:\s*[\+\-\*/]\s*\d+)+)", text)
    if match:
        return match.group(1)
    return None


def simple_agent(user_input: str) -> dict[str, Any]:
    tool_results = []

    # Tool 1: project status
    if any(keyword in user_input for keyword in ["项目状态", "当前项目", "支持哪些"]):
        result = project_status()
        tool_results.append({
            "tool": "project_status",
            "input": None,
            "output": result,
        })

    # Tool 2: calculator
    expression = extract_math_expression(user_input)
    if expression:
        result = calculator(expression)
        tool_results.append({
            "tool": "calculator",
            "input": expression,
            "output": result,
        })

    # Tool 3: RAG search
    if any(keyword in user_input for keyword in ["LangGraph", "LangChain", "Deep Agents", "MCP", "Agent", "RAG"]):
        result = rag_search(user_input)
        tool_results.append({
            "tool": "rag_search",
            "input": user_input,
            "output": result,
        })

    if not tool_results:
        final_answer = "我没有判断出需要调用的工具。你可以尝试问项目状态、RAG/Agent 相关问题，或包含一个算式。"
    else:
        final_answer = "Agent 调用了以下工具并整合结果：\n\n"
        for item in tool_results:
            final_answer += f"## Tool: {item['tool']}\n"
            if item["input"]:
                final_answer += f"Input: {item['input']}\n"
            final_answer += f"Output:\n{item['output']}\n\n"

    return {
        "user_input": user_input,
        "tool_results": tool_results,
        "final_answer": final_answer,
    }


def main():
    question = "如果我要构建一个更可控的多步骤 Agent，应该用什么？顺便算一下 12*8+5。"
    result = simple_agent(question)

    print("User input:")
    print(result["user_input"])

    print("\nFinal answer:")
    print(result["final_answer"])


if __name__ == "__main__":
    main()
