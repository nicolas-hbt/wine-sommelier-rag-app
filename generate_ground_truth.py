"""
Generates ground truth Q→doc_id pairs by asking Groq to write a question
that can only be answered by each sampled wine review.
Saves to eval_data/ground_truth.csv.
"""
import os
import sys
import random
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from ingest import load_documents

load_dotenv()

N_SAMPLES = 100
SEED = 42
OUTPUT = Path("eval_data/ground_truth.csv")
MODEL = "llama-3.3-70b-versatile"
REQUEST_DELAY_SECONDS = 2.2
RETRY_DELAYS = (5, 10, 20)


class GroundTruth(BaseModel):
    question: str


SYSTEM_PROMPT = """You are generating evaluation data for a wine Q&A system.
Given a wine review, write ONE question a user might ask that this review directly answers.
The question should be about the wine's taste, pairing, region, price, or rating.
Return a JSON object with a single 'question' field."""


def generate_question(client, doc):
    prompt = f"Wine review:\n{doc['text'][:600]}"
    for attempt, retry_delay in enumerate(RETRY_DELAYS, start=1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=150,
            )
            result = GroundTruth.model_validate_json(response.choices[0].message.content)
            return {"doc_id": doc["doc_id"], "question": result.question}
        except Exception as e:
            if attempt == len(RETRY_DELAYS):
                print(f"  failed for doc {doc['doc_id']}: {e}")
                return None
            if "rate_limit_exceeded" in str(e) or "429" in str(e):
                time.sleep(retry_delay)
            else:
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
    for doc in tqdm(sampled):
        result = generate_question(client, doc)
        if result:
            results.append(result)
        time.sleep(REQUEST_DELAY_SECONDS)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} pairs to {OUTPUT}")


if __name__ == "__main__":
    main()
