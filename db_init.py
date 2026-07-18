import os
import psycopg
from datetime import datetime

DB_TIMEZONE = datetime.now().astimezone().tzinfo


def get_db_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        dbname=os.getenv("POSTGRES_DB", "wine_assistant"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
    )


def init_db():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS wines (
                    id SERIAL PRIMARY KEY,
                    doc_id TEXT NOT NULL UNIQUE,
                    text TEXT NOT NULL,
                    title TEXT,
                    variety TEXT,
                    winery TEXT,
                    country TEXT,
                    province TEXT,
                    points INTEGER,
                    price FLOAT,
                    description TEXT,
                    embedding vector(384)
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS wines_embedding_idx "
                "ON wines USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            )
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    search_method TEXT NOT NULL,
                    prompt_style TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    response_time FLOAT NOT NULL,
                    cost FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER REFERENCES conversations(id),
                    source TEXT NOT NULL,
                    relevance TEXT,
                    explanation TEXT,
                    score INTEGER,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized.")
