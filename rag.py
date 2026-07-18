"""
WineRAG: query rewriting → pgvector search → optional reranking → LLM answer.
"""
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from search import vector_search

MODEL_PATH = Path(__file__).parent / "models" / "Xenova" / "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

CONCISE_TEMPLATE = """You are a knowledgeable wine sommelier. Answer the question using ONLY the wine reviews provided below.
Be concise and direct — 2-3 sentences maximum.

WINE REVIEWS:
{context}

QUESTION: {question}
ANSWER:"""

DETAILED_TEMPLATE = """You are a knowledgeable wine sommelier. Answer the question using ONLY the wine reviews provided below.
Provide a thorough answer that includes specific wine recommendations with tasting notes, regions, and price guidance where available.
5-6 sentences maximum.

WINE REVIEWS:
{context}

QUESTION: {question}
ANSWER:"""

REWRITE_TEMPLATE = """Rewrite the following wine question to be more specific and retrieval-friendly.
Keep it as a single question. Return only the rewritten question, no explanation.

Original question: {question}
Rewritten question:"""

PROMPT_TEMPLATES = {
    "concise": CONCISE_TEMPLATE,
    "detailed": DETAILED_TEMPLATE,
}


@dataclass
class RAGCallRecord:
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
    timestamp: datetime = field(default_factory=datetime.now)


def _groq_client():
    return OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )


def _calc_cost(prompt_tokens, completion_tokens):
    return (prompt_tokens * 0.59 + completion_tokens * 0.79) / 1_000_000


class WineRAG:
    def __init__(
        self,
        search_method="vector",
        prompt_style="concise",
        use_rewrite=True,
        use_rerank=True,
    ):
        self.search_method = search_method
        self.prompt_style = prompt_style
        self.use_rewrite = use_rewrite
        self.use_rerank = use_rerank
        self.client = _groq_client()
        self.last_call: RAGCallRecord = None
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from embedder import Embedder
            self._embedder = Embedder(path=str(MODEL_PATH))
        return self._embedder

    def _rewrite_query(self, question):
        prompt = REWRITE_TEMPLATE.format(question=question)
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()

    def _search(self, query, filters=None):
        embedder = self._get_embedder()
        return vector_search(embedder, query, n=10, filters=filters)

    def _build_context(self, results):
        return "\n\n---\n\n".join(r["text"] for r in results)

    def _build_prompt(self, question, results):
        context = self._build_context(results)
        template = PROMPT_TEMPLATES[self.prompt_style]
        return template.format(context=context, question=question)

    def rag(self, question, filters=None):
        start = time.time()

        search_query = question
        if self.use_rewrite:
            search_query = self._rewrite_query(question)

        results = self._search(search_query, filters=filters)

        if self.use_rerank and len(results) > 1:
            from rerank import rerank
            results = rerank(question, results)

        prompt = self._build_prompt(question, results)

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledgeable wine sommelier. Use only the provided wine reviews to answer.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content.strip()
        usage = response.usage
        elapsed = time.time() - start

        self.last_call = RAGCallRecord(
            question=question,
            answer=answer,
            search_method=self.search_method,
            prompt_style=self.prompt_style,
            model=GROQ_MODEL,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            response_time=elapsed,
            cost=_calc_cost(usage.prompt_tokens, usage.completion_tokens),
        )
        return answer
