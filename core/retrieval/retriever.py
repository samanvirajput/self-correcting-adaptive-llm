from core.embeddings.pipeline import embed_text


def retrieve(
    query: str,
    memory,
    top_k: int = 5,
    min_score: float = 0.3,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    """Semantic search over both memory tiers. Returns ranked list of memory items."""
    qvec = embed_text(query, model_name=embedding_model)
    results = memory.retrieve(qvec, top_k=top_k)
    return [r for r in results if r.get("score", 0) >= min_score]


def format_memories_for_prompt(memories: list[dict], max_items: int = 3) -> str:
    if not memories:
        return ""
    lines = []
    for m in memories[:max_items]:
        content = m.get("content", "")
        score = m.get("score", 0)
        lines.append(f"  [{score:.2f}] {content[:200]}")
    return "\n".join(lines)
