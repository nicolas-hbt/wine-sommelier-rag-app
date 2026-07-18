"""
Search module: pgvector cosine similarity with optional hard metadata filters.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

MODEL_PATH = Path(__file__).parent / "models" / "Xenova" / "all-MiniLM-L6-v2"


def _get_embedder():
    from embedder import Embedder
    return Embedder(path=str(MODEL_PATH))


def vector_search(embedder, query, n=10, filters=None, conn=None):
    """
    Cosine similarity search using pgvector.

    filters: optional dict with keys 'max_price' (float) and/or 'country' (str).
    conn: optional existing psycopg connection; if None, a new one is opened.
    """
    from db_init import get_db_connection

    query_emb = embedder.encode(query, normalize=True).tolist()

    where_clauses = []
    params = []

    if filters:
        if filters.get("max_price") is not None:
            where_clauses.append("price <= %s")
            params.append(filters["max_price"])
        if filters.get("country"):
            where_clauses.append("country ILIKE %s")
            params.append(filters["country"])

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT doc_id, text,
               1 - (embedding <=> %s::vector) AS score
        FROM wines
        {where_sql}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    params = [query_emb] + params + [query_emb, n]

    close_conn = conn is None
    if close_conn:
        conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        if close_conn:
            conn.close()

    return [{"doc_id": row[0], "text": row[1], "score": float(row[2])} for row in rows]
