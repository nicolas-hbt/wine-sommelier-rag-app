"""
Ingestion pipeline: loads the WineEnthusiast CSV, embeds with ONNX MiniLM,
and inserts into PostgreSQL wines table with pgvector embeddings.
"""
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from db_init import get_db_connection, init_db
from embedder import Embedder

DATA_PATH = Path(__file__).parent / "winemag-data-130k-v2.csv"
MODEL_PATH = Path(__file__).parent / "models" / "Xenova" / "all-MiniLM-L6-v2"
EMBED_BATCH = 64


def build_doc_text(row):
    price = f"${row['price']:.0f}" if pd.notna(row["price"]) else "price unknown"
    region = row["province"] if pd.notna(row["province"]) else ""
    country = row["country"] if pd.notna(row["country"]) else ""
    location = f"{country}, {region}".strip(", ")
    designation = f" ({row['designation']})" if pd.notna(row["designation"]) else ""
    header = (
        f"{row['title']}{designation} | {row['variety']} | "
        f"{location} | {row['points']} pts | {price}"
    )
    return f"{header}\n{row['description']}"


def load_documents(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=["description"]).reset_index(drop=True)
    docs = []
    for i, row in df.iterrows():
        docs.append(
            {
                "doc_id": str(i),
                "text": build_doc_text(row),
                "title": str(row["title"]) if pd.notna(row["title"]) else "",
                "variety": str(row["variety"]) if pd.notna(row["variety"]) else "",
                "winery": str(row["winery"]) if pd.notna(row["winery"]) else "",
                "country": str(row["country"]) if pd.notna(row["country"]) else "",
                "province": str(row["province"]) if pd.notna(row["province"]) else "",
                "points": int(row["points"]) if pd.notna(row["points"]) else 0,
                "price": float(row["price"]) if pd.notna(row["price"]) else None,
                "description": str(row["description"]),
            }
        )
    return docs


def build_pgvector_index(docs, embedder):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE wines RESTART IDENTITY")
        conn.commit()

        texts = [d["text"] for d in docs]
        print(f"Embedding {len(texts)} documents in batches of {EMBED_BATCH}...")
        all_embeddings = []
        for i in tqdm(range(0, len(texts), EMBED_BATCH)):
            batch = texts[i : i + EMBED_BATCH]
            embs = embedder.encode_batch(batch, normalize=True)
            all_embeddings.extend(embs.tolist())

        print("Inserting into PostgreSQL...")
        insert_sql = """
            INSERT INTO wines
                (doc_id, text, title, variety, winery, country, province,
                 points, price, description, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding
        """
        batch_size = 500
        for i in tqdm(range(0, len(docs), batch_size)):
            batch_docs = docs[i : i + batch_size]
            batch_embs = all_embeddings[i : i + batch_size]
            rows = [
                (
                    d["doc_id"], d["text"], d["title"], d["variety"], d["winery"],
                    d["country"], d["province"], d["points"], d["price"],
                    d["description"], emb,
                )
                for d, emb in zip(batch_docs, batch_embs)
            ]
            with conn.cursor() as cur:
                cur.executemany(insert_sql, rows)
            conn.commit()
    finally:
        conn.close()

    print(f"Inserted {len(docs)} wine documents into PostgreSQL.")


def main():
    print("Initializing database schema...")
    init_db()

    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.")

    if not MODEL_PATH.exists():
        print("Downloading ONNX embedding model...")
        from download_model import download
        download("Xenova/all-MiniLM-L6-v2", dest=str(MODEL_PATH.parent.parent))

    embedder = Embedder(path=str(MODEL_PATH))

    print("Building pgvector index...")
    build_pgvector_index(docs, embedder)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
