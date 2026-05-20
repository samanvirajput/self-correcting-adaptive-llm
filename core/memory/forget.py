import os
import json
from datetime import datetime, timezone

from core.correction.engine import CORRECTIONS_DIR


def decay_score(memory_item: dict) -> float:
    """Score in [0, 1]. Lower = more forgettable."""
    now = datetime.now(timezone.utc)
    created_str = memory_item.get("created_at", "")
    try:
        created = datetime.fromisoformat(created_str)
        age_days = (now - created).total_seconds() / 86400
    except Exception:
        age_days = 30.0

    recency = max(0.0, 1.0 - age_days / 90.0)
    access = min(1.0, memory_item.get("access_count", 0) / 10.0)
    confidence = float(memory_item.get("confidence", 0.5))
    return round(0.5 * recency + 0.3 * access + 0.2 * confidence, 4)


def prune_low_value(user_id: str, memory, threshold: float = 0.2) -> int:
    """Remove hot-tier memories whose decay_score < threshold. Returns pruned count."""
    pruned = 0
    survivors_meta = []
    survivors_vecs = []
    for i, item in enumerate(memory._hot_meta):
        if item.get("user_id") == user_id and decay_score(item) < threshold:
            pruned += 1
        else:
            survivors_meta.append(item)
            if i < len(memory._hot_vectors):
                survivors_vecs.append(memory._hot_vectors[i])

    memory._hot_meta = survivors_meta
    memory._hot_vectors = survivors_vecs

    try:
        import faiss
        memory._index.reset()
        if survivors_vecs:
            import numpy as np
            memory._index.add(np.stack(survivors_vecs, axis=0))
    except Exception:
        pass

    return pruned


def explicit_forget(user_id: str, memory_id: str, memory) -> bool:
    """User-triggered deletion by memory id from both tiers."""
    found = False
    new_meta, new_vecs = [], []
    for i, item in enumerate(memory._hot_meta):
        if item.get("id") == memory_id:
            found = True
        else:
            new_meta.append(item)
            if i < len(memory._hot_vecs):
                new_vecs.append(memory._hot_vectors[i])
    memory._hot_meta = new_meta
    memory._hot_vectors = new_vecs

    try:
        memory._collection.delete(ids=[memory_id])
        found = True
    except Exception:
        pass

    return found


def contradiction_forget(
    user_id: str,
    old_pattern_id: str,
    new_description: str,
    corrections_dir: str = CORRECTIONS_DIR,
) -> bool:
    """Replace an old correction pattern file with updated description."""
    for fname in os.listdir(corrections_dir):
        if not fname.startswith(user_id):
            continue
        path = os.path.join(corrections_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("id") == old_pattern_id:
                data["description"] = new_description
                data["created_at"] = datetime.now(timezone.utc).isoformat()
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
                return True
        except Exception:
            pass
    return False
