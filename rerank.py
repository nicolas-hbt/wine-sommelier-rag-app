"""
Local ONNX cross-encoder reranker using BAAI/bge-reranker-base.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_PATH = Path(__file__).parent / "models" / "BAAI" / "bge-reranker-base"

_session = None
_tokenizer = None


def _load_model():
    global _session, _tokenizer
    if _session is None:
        _tokenizer = Tokenizer.from_file(str(MODEL_PATH / "tokenizer.json"))
        _session = ort.InferenceSession(
            str(MODEL_PATH / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
    return _session, _tokenizer


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rerank(question, results):
    """
    Score each result against the question using the cross-encoder.
    Returns results sorted by relevance score descending.
    """
    if not results:
        return results

    session, tokenizer = _load_model()
    input_names = {inp.name for inp in session.get_inputs()}

    pairs = [f"{question} [SEP] {r['text'][:512]}" for r in results]

    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=512)
    encoded = tokenizer.encode_batch(pairs)

    feed = {}
    if "input_ids" in input_names:
        feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
    if "attention_mask" in input_names:
        feed["attention_mask"] = np.array([e.attention_mask for e in encoded], dtype=np.int64)
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)

    logits = session.run(None, feed)[0]
    # logits shape: (batch, 1) or (batch,) — flatten and sigmoid
    scores = _sigmoid(logits.reshape(-1))

    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked]
