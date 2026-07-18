"""
Generates ground truth Q→doc_id pairs by asking Groq to write a question
that can only be answered by each sampled wine review.
Saves to eval_data/ground_truth.csv.
"""
import os
import sys
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from ingest import load_documents

load_dotenv()

N_SAMPLES = 500
SEED = 42
OUTPUT = Path("eval_data/ground_truth.csv")
MODEL = "llama-3.3-70b-versatile"


class GroundTruth(BaseModel):
    question: str


SYSTEM_PROMPT = """You are generating evaluation data for a wine Q&A system.
Given a wine review, write ONE question a user might ask that this review directly answers.
The question should be about the wine's taste, pairing, region, price, or rating.
Return a JSON object with a single 'question' field."""


def generate_question(client, doc):
    prompt = f"Wine review:\n{doc['text'][:600]}"
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ground_truth",
                        "schema": GroundTruth.model_json_schema(),
                    },
                },
                max_tokens=150,
            )
            result = GroundTruth.model_validate_json(response.choices[0].message.content)
            return {"doc_id": doc["doc_id"], "question": result.question}
        except Exception as e:
            if attempt == 2:
                print(f"  failed for doc {doc['doc_id']}: {e}")
                return None
            time.sleep(2 ** attempt)


def main():
    OUTPUT.parent.mkdir(exist_ok=True)
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

    print("Loading documents...")
    docs = load_documents()
    random.seed(SEED)
    sampled = random.sample(docs, min(N_SAMPLES, len(docs)))
    print(f"Generating {len(sampled)} ground truth Q&A pairs...")

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(generate_question, client, doc) for doc in sampled]
        for f in tqdm(futures):
            result = f.result()
            if result:
                results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} pairs to {OUTPUT}")


if __name__ == "__main__":
    main()
