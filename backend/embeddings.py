from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingEngine:
    _instance = None

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        # Return existing instance if already initialized
        if cls._instance is None:
            cls._instance = super(EmbeddingEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if getattr(self, "_initialized", False):
            return
        print(f"🔹 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✅ Embedding model ready")
        self._initialized = True

    def embed(self, text: str) -> list:
        return self.model.encode(text).tolist()

    @staticmethod
    def cosine_similarity(vec1, vec2):
        v1, v2 = np.array(vec1), np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
