from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from src.models.enums import WordSource
from src.utils.text_norm import normalize_vocab_key


@dataclass(frozen=True)
class ExerciseSelectionConfig:
    total_limit: int = 12
    uncertain_limit: int = 5
    new_limit: int = 4
    russian_limit: int = 2
    translator_limit: int = 3
    allow_phrases: bool = False
    max_mastery_for_exercises: int = 80


@dataclass(frozen=True)
class ExerciseWord:
    item: str
    item_type: str
    priority: int
    reason: str
    source: str
    difficulty_level_cefr: Optional[str] = None
    example: Optional[dict] = None


def select_exercise_words(
    *,
    lesson_vocabulary: dict[str, Any],
    user_vocab: dict[str, dict[str, Any]],
    translator_words: Optional[Iterable[dict[str, Any]]] = None,
    config: Optional[ExerciseSelectionConfig] = None,
) -> List[ExerciseWord]:
    cfg = config or ExerciseSelectionConfig()

    picks: List[ExerciseWord] = []
    seen: set[str] = set()

    def can_use_item(item: str, item_type: str) -> bool:
        if not item:
            return False
        if not cfg.allow_phrases and " " in item:
            return False
        if item_type == "PHRASE" and not cfg.allow_phrases:
            return False
        before = user_vocab.get(item, {})
        mastery = int(before.get("mastery", 0) or 0)
        status = str(before.get("status", "new") or "new")
        if status == "known" or mastery >= cfg.max_mastery_for_exercises:
            return False
        return True

    def add_candidate(
        *,
        item: str,
        item_type: str,
        priority: int,
        reason: str,
        source: str,
        difficulty_level_cefr: Optional[str] = None,
        example: Optional[dict] = None,
    ) -> None:
        key = normalize_vocab_key(item)
        if not key or key in seen:
            return
        if not can_use_item(key, item_type):
            return
        seen.add(key)
        picks.append(
            ExerciseWord(
                item=key,
                item_type=item_type,
                priority=priority,
                reason=reason,
                source=source,
                difficulty_level_cefr=difficulty_level_cefr,
                example=example,
            )
        )

    def add_from_block(
        rows: Iterable[dict[str, Any]],
        *,
        limit: int,
        priority: int,
        reason: str,
        source: str,
    ) -> None:
        count = 0
        for row in rows:
            if count >= limit:
                break
            item = str(row.get("item", ""))
            item_type = str(row.get("item_type", "WORD"))
            add_candidate(
                item=item,
                item_type=item_type,
                priority=priority,
                reason=reason,
                source=source,
                difficulty_level_cefr=row.get("difficulty_level_cefr"),
                example=row.get("example"),
            )
            if normalize_vocab_key(item) in seen:
                count += 1

    add_from_block(
        lesson_vocabulary.get("uncertain_top20", []) or [],
        limit=cfg.uncertain_limit,
        priority=100,
        reason="low_confidence_or_pause",
        source="lesson_uncertain",
    )
    add_from_block(
        lesson_vocabulary.get("new_candidates", []) or [],
        limit=cfg.new_limit,
        priority=80,
        reason="new_usage",
        source="lesson_new",
    )
    add_from_block(
        lesson_vocabulary.get("russian_used", []) or [],
        limit=cfg.russian_limit,
        priority=70,
        reason="needs_translation",
        source="lesson_russian",
    )

    translator_rows = translator_words or _translator_rows_from_user_vocab(user_vocab)
    add_from_translator(
        translator_rows,
        seen=seen,
        picks=picks,
        cfg=cfg,
    )

    picks.sort(key=lambda r: (-r.priority, r.item))
    return picks[: cfg.total_limit]


def _translator_rows_from_user_vocab(user_vocab: dict[str, dict[str, Any]]) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    for key, row in user_vocab.items():
        sources = row.get("source") or []
        if WordSource.YANDEX_TRANSLATOR.value not in sources:
            continue
        rows.append({
            "item": key,
            "item_type": row.get("item_type", "WORD"),
            "difficulty_level_cefr": row.get("difficulty_level_cefr"),
            "translation_text": row.get("translation_text"),
            "translation_lemma": row.get("translation_lemma"),
        })
    return rows


def add_from_translator(
    rows: Iterable[dict[str, Any]],
    *,
    seen: set[str],
    picks: List[ExerciseWord],
    cfg: ExerciseSelectionConfig,
) -> None:
    count = 0
    for row in rows:
        if count >= cfg.translator_limit:
            break
        item = str(row.get("item", ""))
        item_type = str(row.get("item_type", "WORD"))
        key = normalize_vocab_key(item)
        if not key or key in seen:
            continue
        if not cfg.allow_phrases and " " in key:
            continue
        seen.add(key)
        picks.append(
            ExerciseWord(
                item=key,
                item_type=item_type,
                priority=60,
                reason="translator_history",
                source=WordSource.YANDEX_TRANSLATOR.value,
                difficulty_level_cefr=row.get("difficulty_level_cefr"),
                example=None,
            )
        )
        count += 1
