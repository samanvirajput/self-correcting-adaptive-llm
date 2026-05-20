import os

CONTEXT_LIMIT = int(os.getenv("CONTEXT_LIMIT", "2048"))
_APPROX_CHARS_PER_TOKEN = 4

_session_token_counts: dict[str, int] = {}


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // _APPROX_CHARS_PER_TOKEN)


def _truncate_to_limit(prompt: str, max_tokens: int) -> str:
    if _approx_tokens(prompt) <= max_tokens:
        return prompt
    # Trim from the front (oldest context), preserve the last max_tokens worth
    char_limit = max_tokens * _APPROX_CHARS_PER_TOKEN
    return "...[context truncated]\n" + prompt[-char_limit:]


def generate(prompt: str, model, max_tokens: int = 512, session_id: str = "default") -> str:
    safe_prompt = _truncate_to_limit(prompt, CONTEXT_LIMIT - max_tokens)
    result = model.generate(safe_prompt, max_tokens=max_tokens)
    used = _approx_tokens(prompt) + _approx_tokens(result)
    _session_token_counts[session_id] = _session_token_counts.get(session_id, 0) + used
    return result


def get_session_token_count(session_id: str = "default") -> int:
    return _session_token_counts.get(session_id, 0)


def reset_session(session_id: str = "default") -> None:
    _session_token_counts.pop(session_id, None)
