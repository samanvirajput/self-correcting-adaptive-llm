import os
import uuid
import json
import numpy as np
from datetime import datetime, timezone

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

import chromadb

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")
_HOT_MAX = 200  # entries before promoting oldest to cold


class LongTermMemory:
    """Dual-tier memory: FAISS hot (in-memory) + ChromaDB cold (persistent)."""

    def __init__(self, user_id: str, dim: int = 384, chroma_path: str = CHROMA_PATH):
        self.user_id = user_id
        self.dim = dim
        self.chroma_path = chroma_path

        # --- HOT TIER (FAISS) ---
        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(dim)  # inner product on L2-normalised = cosine
        else:
            self._index = None
        self._hot_vectors: list[np.ndarray] = []
        self._hot_meta: list[dict] = []

        # --- COLD TIER (ChromaDB) ---
        os.makedirs(chroma_path, exist_ok=True)
        self._client = chromadb.PersistentClient(path=chroma_path)
        safe_name = f"ltm_{user_id.replace('-', '_')}"
        self._collection = self._client.get_or_create_collection(safe_name)

    # ------------------------------------------------------------------ store

    def store(
        self,
        content: str,
        embedding: list[float],
        metadata: dict | None = None,
        tier: str = "hot",
    ) -> str:
        mem_id = str(uuid.uuid4())[:12]
        meta = {
            "id": mem_id,
            "user_id": self.user_id,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "access_count": 0,
            **(metadata or {}),
        }

        if tier == "hot":
            vec = np.array(embedding, dtype="float32")
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self._hot_vectors.append(vec)
            self._hot_meta.append(meta)
            if _FAISS_AVAILABLE and self._index is not None:
                self._index.add(vec.reshape(1, -1))
            if len(self._hot_meta) >= _HOT_MAX:
                self.promote_to_cold()
        else:
            self._cold_store(mem_id, content, embedding, meta)

        return mem_id

    def _cold_store(self, mem_id: str, content: str, embedding: list[float], meta: dict):
        self._collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[{k: str(v) for k, v in meta.items()}],
            ids=[mem_id],
        )

    # --------------------------------------------------------------- retrieve

    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        results: list[dict] = []

        # --- HOT TIER ---
        if self._hot_vectors and _FAISS_AVAILABLE and self._index is not None and self._index.ntotal > 0:
            qvec = np.array(query_embedding, dtype="float32")
            norm = np.linalg.norm(qvec)
            if norm > 0:
                qvec = qvec / norm
            k = min(top_k, self._index.ntotal)
            scores, idxs = self._index.search(qvec.reshape(1, -1), k)
            for score, idx in zip(scores[0], idxs[0]):
                if idx >= 0 and idx < len(self._hot_meta):
                    item = dict(self._hot_meta[idx])
                    item["score"] = float(score)
                    item["tier"] = "hot"
                    results.append(item)

        elif self._hot_vectors and not _FAISS_AVAILABLE:
            # Brute-force cosine without FAISS
            qvec = np.array(query_embedding, dtype="float32")
            for i, vec in enumerate(self._hot_vectors):
                score = float(np.dot(qvec, vec) / (np.linalg.norm(qvec) * np.linalg.norm(vec) + 1e-9))
                item = dict(self._hot_meta[i])
                item["score"] = score
                item["tier"] = "hot"
                results.append(item)
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

        # --- COLD TIER fallback ---
        if len(results) < top_k:
            need = top_k - len(results)
            try:
                cold = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=need,
                )
                docs = cold.get("documents", [[]])[0]
                metas = cold.get("metadatas", [[]])[0]
                dists = cold.get("distances", [[]])[0]
                for doc, meta, dist in zip(docs, metas, dists):
                    item = dict(meta)
                    item["content"] = doc
                    item["score"] = round(1 - float(dist), 4)
                    item["tier"] = "cold"
                    results.append(item)
            except Exception:
                pass

        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    # ------------------------------------------------------- promote_to_cold

    def promote_to_cold(self, keep_recent: int = 50) -> int:
        if len(self._hot_meta) <= keep_recent:
            return 0
        to_promote = self._hot_meta[:-keep_recent]
        vecs_to_promote = self._hot_vectors[:-keep_recent]
        promoted = 0
        for meta, vec in zip(to_promote, vecs_to_promote):
            try:
                self._cold_store(meta["id"], meta["content"], vec.tolist(), meta)
                promoted += 1
            except Exception:
                pass
        self._hot_meta = self._hot_meta[-keep_recent:]
        self._hot_vectors = self._hot_vectors[-keep_recent:]
        if _FAISS_AVAILABLE and self._index is not None:
            self._index.reset()
            if self._hot_vectors:
                mat = np.stack(self._hot_vectors, axis=0)
                self._index.add(mat)
        return promoted

    # ------------------------------------------------------------------- util

    def count(self) -> dict:
        cold_count = self._collection.count()
        return {"hot": len(self._hot_meta), "cold": cold_count, "total": len(self._hot_meta) + cold_count}
