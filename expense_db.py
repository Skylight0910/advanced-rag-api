import sqlite3
from pathlib import Path

DB_PATH = Path("expenses.db")


def init_database() -> None:
    connection = sqlite3.connect(DB_PATH)

    try:
        connection.executescript("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expense_date TEXT NOT NULL,
            category TEXT NOT NULL,
            amount REAL NOT NULL,
            description TEXT NOT NULL,
            payment_method TEXT NOT NULL
        );

        DELETE FROM expenses;

        INSERT INTO expenses
            (expense_date, category, amount, description, payment_method)
        VALUES
            ('2026-05-28', '餐饮', 28.00, '午餐', '微信'),
            ('2026-05-29', '交通', 15.00, '打车', '支付宝'),
            ('2026-06-01', '餐饮', 42.50, '晚餐', '微信'),
            ('2026-06-01', '数码', 199.00, '键盘', '支付宝'),
            ('2026-06-02', '交通', 4.00, '地铁', '微信'),
            ('2026-06-02', '餐饮', 18.00, '早餐', '微信');
        """)

        connection.commit()
    finally:
        connection.close()


def sql_query(query: str) -> list[dict]:
    cleaned_query = query.strip()

    if cleaned_query.endswith(";"):
        cleaned_query = cleaned_query[:-1].strip()

    normalized = cleaned_query.lower()

    if not normalized.startswith(("select ", "with ")):
        raise ValueError("只允许执行 SELECT 或 WITH 查询")

    if ";" in cleaned_query:
        raise ValueError("每次只允许执行一条 SQL")

    connection = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row

    try:
        rows = connection.execute(cleaned_query).fetchall()
        return [dict(row) for row in rows[:20]]
    finally:
        connection.close()


def main():
    init_database()

    questions = [
        """
        SELECT SUM(amount) AS total
        FROM expenses
        WHERE category = '餐饮';
        """,
        """
        SELECT category, SUM(amount) AS total
        FROM expenses
        GROUP BY category
        ORDER BY total DESC;
        """,
        "DELETE FROM expenses;",
    ]

    for query in questions:
        print("=" * 80)
        print("SQL:", query.strip())

        try:
            print("Result:", sql_query(query))
        except Exception as error:
            print("Blocked:", error)


if __name__ == "__main__":
    main()
