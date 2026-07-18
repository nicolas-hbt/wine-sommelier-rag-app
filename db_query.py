import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db_init import get_db_connection


@dataclass
class ConversationRow:
    id: int
    question: str
    answer: str
    search_method: str
    prompt_style: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime


def get_conversations(limit=100):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question, answer, search_method, prompt_style, model,
                       prompt_tokens, completion_tokens, total_tokens,
                       response_time, cost, timestamp
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [ConversationRow(*row) for row in rows]


def get_stats():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*),
                    COALESCE(AVG(response_time), 0),
                    COALESCE(SUM(cost), 0),
                    COALESCE(AVG(total_tokens), 0)
                FROM conversations
            """)
            row = cur.fetchone()
    finally:
        conn.close()
    return {
        "total": row[0],
        "avg_response_time": float(row[1]),
        "total_cost": float(row[2]),
        "avg_tokens": float(row[3]),
    }


def get_relevance_stats():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT relevance, COUNT(*)
                FROM feedback
                WHERE source = 'judge'
                GROUP BY relevance
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    return dict(rows)


def get_user_feedback_stats():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COALESCE(SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN score < 0 THEN 1 ELSE 0 END), 0)
                FROM feedback
                WHERE source = 'user'
            """)
            row = cur.fetchone()
    finally:
        conn.close()
    return int(row[0]), int(row[1])


def get_requests_per_day(days=14):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DATE(timestamp) as day, COUNT(*) as count
                FROM conversations
                WHERE timestamp >= NOW() - INTERVAL '%s days'
                GROUP BY day
                ORDER BY day
            """, (days,))
            rows = cur.fetchall()
    finally:
        conn.close()
    return rows


def get_search_method_stats():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT search_method, COUNT(*) as count,
                       AVG(response_time) as avg_time
                FROM conversations
                GROUP BY search_method
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    return rows
