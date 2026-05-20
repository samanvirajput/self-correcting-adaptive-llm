from collections import defaultdict

_sessions: dict[str, list[dict]] = defaultdict(list)


def add_turn(session_id: str, role: str, content: str) -> None:
    _sessions[session_id].append({"role": role, "content": content})


def get_context(session_id: str, max_turns: int = 10) -> list[dict]:
    turns = _sessions[session_id]
    return turns[-max_turns * 2:] if len(turns) > max_turns * 2 else list(turns)


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def session_turn_count(session_id: str) -> int:
    return len(_sessions[session_id]) // 2
