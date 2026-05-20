import os
import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

_PATTERN_TYPES = {"verbosity", "tone", "repetition", "factual_error", "ignored_memory"}

_CATEGORY_MAP = {
    "factual": "factual_error",
    "factual errors": "factual_error",
    "tone": "tone",
    "tone mismatch": "tone",
    "repetition": "repetition",
    "verbosity": "verbosity",
    "ignored context": "ignored_memory",
    "ignored_memory": "ignored_memory",
}

CORRECTIONS_DIR = os.getenv("CORRECTIONS_DIR", "./data/corrections")
os.makedirs(CORRECTIONS_DIR, exist_ok=True)


@dataclass
class CorrectionPattern:
    user_id: str
    pattern_type: str
    description: str
    confidence: float
    created_at: str
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


def detect_patterns(reflection_result, user_id: str) -> list[CorrectionPattern]:
    patterns = []
    for issue in reflection_result.issues:
        raw_cat = issue.get("category", "").lower()
        pattern_type = _CATEGORY_MAP.get(raw_cat, raw_cat)
        if pattern_type not in _PATTERN_TYPES:
            pattern_type = "factual_error"
        confidence = min(1.0, 0.5 + 0.1 * len(reflection_result.issues))
        patterns.append(CorrectionPattern(
            user_id=user_id,
            pattern_type=pattern_type,
            description=issue.get("description", ""),
            confidence=round(confidence, 3),
            created_at=datetime.now(timezone.utc).isoformat(),
        ))
    return patterns


def log_correction(pattern: CorrectionPattern, corrections_dir: str = CORRECTIONS_DIR) -> None:
    path = os.path.join(corrections_dir, f"{pattern.user_id}_{pattern.id}.json")
    with open(path, "w") as f:
        json.dump(asdict(pattern), f, indent=2)


def get_active_corrections(user_id: str, corrections_dir: str = CORRECTIONS_DIR) -> list[CorrectionPattern]:
    patterns = []
    if not os.path.isdir(corrections_dir):
        return patterns
    for fname in os.listdir(corrections_dir):
        if not fname.startswith(user_id) or not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(corrections_dir, fname)) as f:
                data = json.load(f)
            patterns.append(CorrectionPattern(**data))
        except Exception:
            pass
    patterns.sort(key=lambda p: p.created_at, reverse=True)
    return patterns[:20]
