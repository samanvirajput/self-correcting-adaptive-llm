from sentence_transformers import SentenceTransformer
import torch

class SentenceTransformerWrapper:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # ✅ Detect the proper device (MPS for Apple Silicon, else CPU)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🔹 Loading embedding model: {model_name} (device={device})")
        self.model = SentenceTransformer(model_name, device=device)
        print("✅ Embedding model ready")

    def embed_documents(self, texts):
        """Embed a list of documents (used by LangChain)."""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        """Embed a single query (used by LangChain retriever)."""
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()
