import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db_init import get_db_connection, DB_TIMEZONE


def save_conversation(record):
    timestamp = datetime.now(DB_TIMEZONE)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (
                    question, answer, search_method, prompt_style, model,
                    prompt_tokens, completion_tokens, total_tokens,
                    response_time, cost, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    record.question,
                    record.answer,
                    record.search_method,
                    record.prompt_style,
                    record.model,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.response_time,
                    record.cost,
                    timestamp,
                ),
            )
            conversation_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return conversation_id


def save_feedback(conversation_id, source, relevance=None, explanation=None, score=None):
    timestamp = datetime.now(DB_TIMEZONE)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (
                    conversation_id, source, relevance, explanation, score, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (conversation_id, source, relevance, explanation, score, timestamp),
            )
        conn.commit()
    finally:
        conn.close()
