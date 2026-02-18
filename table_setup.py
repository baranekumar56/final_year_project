import psycopg2
import psycopg2.extras
import random
import math

DB_DSN = "postgresql://postgres:postgres@localhost:5432/final_year_project"  # change this

TOTAL_ROWS = 20000000
BATCH_SIZE = 200000
MAX_COUNTRY_ID = 100  # country ids will be in [0, MAX_COUNTRY_ID)


def main():
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    cur = conn.cursor()

    # Drop and recreate table
    cur.execute("DROP TABLE IF EXISTS sales;")
    cur.execute(
        """
        CREATE TABLE sales (
            id SERIAL PRIMARY KEY,
            country INT NOT NULL,
            amount REAL NOT NULL
        );
        """
    )

    print("Inserting random data...")

    remaining = TOTAL_ROWS
    while remaining > 0:
        n = min(BATCH_SIZE, remaining)
        remaining -= n

        rows = []
        for _ in range(n):
            country = random.randint(0, MAX_COUNTRY_ID)  # uniform over countries
            # amount: mix of small and large values, with more weight near 0–500
            base = random.random()  # 0..1
            # skew distribution: square to bias low, then scale
            amount = (base ** 2) * 1000.0
            rows.append((country, amount))

        # Use execute_values for efficient multi-row insert.[web:134][web:136]
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO sales (country, amount) VALUES %s",
            rows,
            template=None,
            page_size=BATCH_SIZE,
        )

    print(f"Inserted {TOTAL_ROWS} random rows into 'sales'.")

    # Sanity check with your query
    cur.execute(
        """
        SELECT country, SUM(amount)
        FROM sales
        WHERE amount > 100
        GROUP BY country
        ORDER BY country
        LIMIT 10;
        """
    )
    print("Sample results (first 10 countries):")
    for row in cur.fetchall():
        print(row)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
