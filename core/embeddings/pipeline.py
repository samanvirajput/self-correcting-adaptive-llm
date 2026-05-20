import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL: SentenceTransformer | None = None
_MODEL_NAME: str = ""


def _get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _MODEL, _MODEL_NAME
    if _MODEL is None or _MODEL_NAME != model_name:
        _MODEL = SentenceTransformer(model_name)
        _MODEL_NAME = model_name
    return _MODEL


def embed_text(text: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    return _get_model(model_name).encode(text, convert_to_numpy=True).tolist()


def embed_batch(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> list[list[float]]:
    return _get_model(model_name).encode(texts, convert_to_numpy=True).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0
