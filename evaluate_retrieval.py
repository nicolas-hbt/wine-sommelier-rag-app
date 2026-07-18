"""
Evaluates pgvector search using Hit Rate @5 and MRR @5.
Reads eval_data/ground_truth.csv and prints results.
"""
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from embedder import Embedder
from search import vector_search

MODEL_PATH = Path(__file__).parent / "models" / "Xenova" / "all-MiniLM-L6-v2"
GT_PATH = Path("eval_data/ground_truth.csv")


def hit_rate(results, doc_id):
    return int(any(r["doc_id"] == doc_id for r in results))


def reciprocal_rank(results, doc_id):
    for i, r in enumerate(results):
        if r["doc_id"] == doc_id:
            return 1.0 / (i + 1)
    return 0.0


def main():
    gt_df = pd.read_csv(GT_PATH)
    print(f"Loaded {len(gt_df)} ground truth pairs.")

    embedder = Embedder(path=str(MODEL_PATH))

    hits, rrs = [], []
    for _, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc="vector"):
        results = vector_search(embedder, row["question"], n=5)
        hits.append(hit_rate(results, str(row["doc_id"])))
        rrs.append(reciprocal_rank(results, str(row["doc_id"])))

    hr = sum(hits) / len(hits)
    mrr = sum(rrs) / len(rrs)
    print(f"\nvector: hit_rate@5={hr:.3f}, mrr@5={mrr:.3f}")

    results_df = pd.DataFrame([{"method": "vector", "hit_rate@5": hr, "mrr@5": mrr}])
    results_df.to_csv("eval_data/retrieval_results.csv", index=False)


if __name__ == "__main__":
    main()
