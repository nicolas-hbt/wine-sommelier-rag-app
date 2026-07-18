"""
LLM-as-judge: evaluates RAG answer relevance using Groq + Pydantic structured output.
"""
import os
import time

from openai import OpenAI
from pydantic import BaseModel
from typing import Literal


class RelevanceVerdict(BaseModel):
    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    explanation: str


JUDGE_SYSTEM = """You are an expert evaluator for a wine Q&A RAG system.
Analyze the relevance of the generated answer to the given question.

Classify the answer as:
- RELEVANT: the answer directly addresses the question with specific wine recommendations or information
- PARTLY_RELEVANT: the answer partially addresses the question but is incomplete or off-topic in parts
- NON_RELEVANT: the answer does not address the question"""


def _groq_client():
    return OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )


JUDGE_MODEL = "openai/gpt-oss-120b"


def _judge_once(client, question, answer, model=JUDGE_MODEL):
    prompt = f"Question: {question}\n\nGenerated Answer: {answer}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    return RelevanceVerdict.model_validate_json(response.choices[0].message.content)


def evaluate_relevance(question, answer, client=None, max_retries=3):
    if client is None:
        client = _groq_client()
    for attempt in range(max_retries):
        try:
            result = _judge_once(client, question, answer)
            return result.relevance, result.explanation
        except Exception:
            if attempt == max_retries - 1:
                return "NON_RELEVANT", "evaluation failed"
            time.sleep(2 ** attempt)
