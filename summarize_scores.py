import csv
from collections import defaultdict

SCORE_FILE = "eval_scores.csv"

METRICS = [
    "retrieval_relevance",
    "answer_correctness",
    "groundedness",
    "completeness",
]


def main():
    grouped = defaultdict(list)

    with open(SCORE_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            endpoint = row["endpoint"]
            grouped[endpoint].append(row)

    print("| 接口 | 样本数 | retrieval_relevance | answer_correctness | groundedness | completeness | overall |")
    print("|---|---:|---:|---:|---:|---:|---:|")

    for endpoint, rows in grouped.items():
        averages = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if row[metric]]
            averages[metric] = sum(values) / len(values) if values else 0

        overall = sum(averages.values()) / len(METRICS)

        print(
            f"| `{endpoint}` | {len(rows)} | "
            f"{averages['retrieval_relevance']:.2f} | "
            f"{averages['answer_correctness']:.2f} | "
            f"{averages['groundedness']:.2f} | "
            f"{averages['completeness']:.2f} | "
            f"{overall:.2f} |"
        )


if __name__ == "__main__":
    main()
