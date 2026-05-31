import json
import time
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
QUESTION_FILE = Path("eval_questions.json")
OUTPUT_FILE = Path("eval_results.json")

ENDPOINTS = [
    "/ask",
    "/ask_multi_query",
    "/ask_hyde",
    "/ask_rerank",
    "/ask_hybrid",
    "/ask_hybrid_rerank",
]


def load_questions():
    with QUESTION_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def call_endpoint(endpoint: str, query: str):
    url = f"{BASE_URL}{endpoint}"
    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            json={"query": query},
            timeout=120,
        )
        latency = round(time.perf_counter() - start, 3)

        return {
            "ok": response.ok,
            "status_code": response.status_code,
            "latency_seconds": latency,
            "data": response.json() if response.content else None,
            "error": None,
        }
    except Exception as e:
        latency = round(time.perf_counter() - start, 3)
        return {
            "ok": False,
            "status_code": None,
            "latency_seconds": latency,
            "data": None,
            "error": str(e),
        }


def main():
    questions = load_questions()
    results = []

    for item in questions:
        question_id = item["id"]
        query = item["query"]
        expected_points = item.get("expected_points", [])
        category = item.get("category")

        print(f"\n== {question_id}: {query} ==")

        for endpoint in ENDPOINTS:
            print(f"Calling {endpoint} ...")
            result = call_endpoint(endpoint, query)

            results.append({
                "id": question_id,
                "query": query,
                "category": category,
                "expected_points": expected_points,
                "endpoint": endpoint,
                **result,
            })

            print(
                f"{endpoint} -> ok={result['ok']} "
                f"status={result['status_code']} "
                f"latency={result['latency_seconds']}s"
            )

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
