import os
import chromadb
from backend.embeddings import EmbeddingEngine
import random
class VectorMemory:
    def __init__(self, persist_directory: str = "./data/chroma_store"):
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_dir = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Use our own embedding engine for consistency
        self.embedder = EmbeddingEngine()

        # Create or get Chroma collection with a custom embedding function
        self.collection = self.client.get_or_create_collection(name="memory")

    def add_entry(self, query: str, response: str):
        """Store a query-response pair with precomputed embeddings."""
        doc_text = f"User: {query}\nAssistant: {response}"
        doc_id = str(abs(hash(doc_text)))[:16]
        embedding = self.embedder.embed(doc_text)

        self.collection.add(
            documents=[doc_text],
            embeddings=[embedding],
            ids=[doc_id],
        )
        if random.random() < 0.05:  # only 5% of the time
            print("💾 Context updated.")

    def search(self, query: str, top_k: int = 3):
        """Perform semantic search using custom embeddings."""
        query_emb = self.embedder.embed(query)
        results = self.collection.query(query_embeddings=[query_emb], n_results=top_k)

        if not results["documents"]:
            return []
        return [
            {"text": doc, "score": round(float(score), 3)}
            for doc, score in zip(results["documents"][0], results["distances"][0])
        ]
