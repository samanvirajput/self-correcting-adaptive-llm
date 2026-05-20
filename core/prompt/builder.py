_SYSTEM = """You are an adaptive personal AI assistant that learns from every conversation.
You run entirely on-device — no cloud, no external APIs.
You remember past conversations and adapt your style based on explicit user corrections."""


def build_prompt(
    query: str,
    short_term_context: list[dict],
    retrieved_memories: list[dict],
    active_corrections: list,
    user_prefs: dict | None = None,
) -> str:
    parts: list[str] = [_SYSTEM]

    # --- Active correction rules ---
    if active_corrections:
        rules = []
        seen_types: set[str] = set()
        for p in active_corrections[:5]:
            pt = p.pattern_type
            if pt not in seen_types:
                rules.append(f"  - [{pt}] {p.description}")
                seen_types.add(pt)
        if rules:
            parts.append("Behavioral corrections from past interactions:\n" + "\n".join(rules))

    # --- Long-term memory context ---
    if retrieved_memories:
        mem_lines = []
        for m in retrieved_memories[:3]:
            content = m.get("content", "")
            score = m.get("score", 0)
            mem_lines.append(f"  [{score:.2f}] {content[:250]}")
        parts.append("Relevant memories from past sessions:\n" + "\n".join(mem_lines))

    # --- Short-term conversation history ---
    if short_term_context:
        hist_lines = []
        for turn in short_term_context[-10:]:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")[:300]
            hist_lines.append(f"{role}: {content}")
        parts.append("Recent conversation:\n" + "\n".join(hist_lines))

    parts.append(f"User: {query}\nAssistant:")
    return "\n\n".join(parts)
