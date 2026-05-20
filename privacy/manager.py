"""
privacy/manager.py — All operations are local only. No network calls ever.
"""
import os
import json
from datetime import datetime, timezone

from core.correction.engine import CORRECTIONS_DIR, get_active_corrections


def export_user_data(user_id: str, memory=None) -> dict:
    """Return all memories + corrections for a user as a serialisable dict."""
    result: dict = {
        "user_id": user_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "corrections": [],
        "hot_memories": [],
        "cold_memories": [],
        "reflection_logs": [],
    }

    # Corrections
    for p in get_active_corrections(user_id, CORRECTIONS_DIR):
        result["corrections"].append({
            "id": p.id,
            "type": p.pattern_type,
            "description": p.description,
            "confidence": p.confidence,
            "created_at": p.created_at,
        })

    # Memory tiers
    if memory is not None:
        for item in memory._hot_meta:
            if item.get("user_id") == user_id:
                result["hot_memories"].append(item)
        try:
            cold = memory._collection.get(where={"user_id": user_id})
            for doc, meta in zip(cold.get("documents", []), cold.get("metadatas", [])):
                result["cold_memories"].append({"content": doc, **meta})
        except Exception:
            pass

    # Reflection logs
    reflect_dir = "./data/reflection_data"
    if os.path.isdir(reflect_dir):
        for fname in sorted(os.listdir(reflect_dir)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(reflect_dir, fname)) as f:
                    result["reflection_logs"].append(json.load(f))
            except Exception:
                pass

    return result


def delete_user_data(user_id: str, memory=None) -> dict:
    """Full wipe of all user data across all storage locations."""
    counts: dict = {"corrections": 0, "hot_memories": 0, "cold_memories": 0, "reflection_logs": 0}

    # Corrections
    if os.path.isdir(CORRECTIONS_DIR):
        for fname in list(os.listdir(CORRECTIONS_DIR)):
            if fname.startswith(user_id):
                os.remove(os.path.join(CORRECTIONS_DIR, fname))
                counts["corrections"] += 1

    # Memory tiers
    if memory is not None:
        new_meta, new_vecs = [], []
        for i, item in enumerate(memory._hot_meta):
            if item.get("user_id") == user_id:
                counts["hot_memories"] += 1
            else:
                new_meta.append(item)
                if i < len(memory._hot_vectors):
                    new_vecs.append(memory._hot_vectors[i])
        memory._hot_meta = new_meta
        memory._hot_vectors = new_vecs
        try:
            import faiss
            memory._index.reset()
            if new_vecs:
                import numpy as np
                memory._index.add(np.stack(new_vecs, axis=0))
        except Exception:
            pass

        try:
            cold = memory._collection.get(where={"user_id": user_id})
            ids = cold.get("ids", [])
            if ids:
                memory._collection.delete(ids=ids)
                counts["cold_memories"] += len(ids)
        except Exception:
            pass

    # Reflection logs (shared, delete all)
    reflect_dir = "./data/reflection_data"
    if os.path.isdir(reflect_dir):
        for fname in list(os.listdir(reflect_dir)):
            if fname.endswith(".json"):
                os.remove(os.path.join(reflect_dir, fname))
                counts["reflection_logs"] += 1

    return counts


def list_memories(user_id: str, memory=None) -> list[dict]:
    items = []
    if memory is None:
        return items
    for item in memory._hot_meta:
        if item.get("user_id") == user_id:
            items.append({"id": item.get("id"), "tier": "hot",
                          "content": item.get("content", "")[:120],
                          "created_at": item.get("created_at", "")})
    try:
        cold = memory._collection.get(where={"user_id": user_id})
        for doc, meta in zip(cold.get("documents", []), cold.get("metadatas", [])):
            items.append({"id": meta.get("id"), "tier": "cold",
                          "content": doc[:120],
                          "created_at": meta.get("created_at", "")})
    except Exception:
        pass
    return items


def forget_memory(user_id: str, memory_id: str, memory=None) -> bool:
    from core.memory.forget import explicit_forget
    if memory is None:
        return False
    return explicit_forget(user_id, memory_id, memory)
