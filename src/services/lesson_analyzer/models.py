from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

Role = Literal["teacher", "student"]

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass
class WordToken:
    text: str
    start: float
    end: float
    prob: Optional[float]
    pause_ms: Optional[int]
    role: Role

@dataclass
class Utterance:
    role: Role
    start: float
    end: float
    text: str
    avg_prob: Optional[float] = None
    words_count: int = 0
