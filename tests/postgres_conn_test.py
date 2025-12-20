import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}


def test_postgres_conn():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            res = cur.execute("""
                              SELECT table_schema, table_name
                              FROM information_schema.tables
                              WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                              ORDER BY table_schema, table_name;
                              """)
