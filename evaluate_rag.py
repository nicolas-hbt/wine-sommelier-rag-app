"""
RAG evaluation: compares CONCISE vs DETAILED prompt templates using LLM-as-judge.
Samples 200 questions from eval_data/ground_truth.csv.
Saves results to eval_data/rag_evaluation_results.csv.
"""
import os
import sys
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from rag import WineRAG
from judge import evaluate_relevance

load_dotenv()

N_SAMPLES = 200
SEED = 42
GT_PATH = Path("eval_data/ground_truth.csv")
OUTPUT = Path("eval_data/rag_evaluation_results.csv")


def evaluate_one(args):
    rag, question, prompt_style = args
    rag_instance = WineRAG(
        search_method="hybrid",
        prompt_style=prompt_style,
        use_rewrite=False,
        use_rerank=False,
    )
    try:
        answer = rag_instance.rag(question)
        relevance, explanation = evaluate_relevance(question, answer)
        return {
            "question": question,
            "answer": answer,
            "prompt_style": prompt_style,
            "relevance": relevance,
            "explanation": explanation,
        }
    except Exception as e:
        return {
            "question": question,
            "answer": "",
            "prompt_style": prompt_style,
            "relevance": "NON_RELEVANT",
            "explanation": str(e),
        }


def main():
    gt_df = pd.read_csv(GT_PATH)
    random.seed(SEED)
    sampled = gt_df.sample(min(N_SAMPLES, len(gt_df)), random_state=SEED)
    questions = sampled["question"].tolist()
    print(f"Evaluating {len(questions)} questions × 2 prompts...")

    tasks = []
    for q in questions:
        tasks.append((None, q, "concise"))
        tasks.append((None, q, "detailed"))

    results = []
    for task in tqdm(tasks):
        result = evaluate_one(task)
        results.append(result)
        time.sleep(0.1)  # be gentle with Groq rate limits

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT, index=False)

    print("\nRAG Evaluation Results:")
    for style in ["concise", "detailed"]:
        sub = df[df["prompt_style"] == style]
        counts = sub["relevance"].value_counts()
        total = len(sub)
        relevant_pct = counts.get("RELEVANT", 0) / total * 100
        partly_pct = counts.get("PARTLY_RELEVANT", 0) / total * 100
        non_pct = counts.get("NON_RELEVANT", 0) / total * 100
        print(
            f"  {style}: RELEVANT={relevant_pct:.1f}%  "
            f"PARTLY={partly_pct:.1f}%  NON={non_pct:.1f}%"
        )

    best = df.groupby("prompt_style").apply(
        lambda g: (g["relevance"] == "RELEVANT").mean()
    ).idxmax()
    print(f"\nBest prompt style: {best}")


if __name__ == "__main__":
    main()
