from __future__ import annotations

from typing import Dict, Any, List

from .models import now_iso


def build_top_problems_from_uncertain(uncertain_top20: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pause_cnt = sum(1 for r in uncertain_top20 if "pause_before" in (r.get("why") or []))
    lowc_cnt = sum(1 for r in uncertain_top20 if "low_confidence" in (r.get("why") or []))

    problems = []
    if pause_cnt:
        ex = next((r for r in uncertain_top20 if "pause_before" in (r.get("why") or [])), None)
        problems.append({
            "problem": "Паузы перед ключевыми словами",
            "count": pause_cnt,
            "example": (ex.get("example") if ex else None),
        })
    if lowc_cnt:
        ex = next((r for r in uncertain_top20 if "low_confidence" in (r.get("why") or [])), None)
        problems.append({
            "problem": "Низкая уверенность распознавания (возможное произношение/дикция)",
            "count": lowc_cnt,
            "example": (ex.get("example") if ex else None),
        })
    return problems[:2]


def build_badges(
        *,
        talk_metrics: Dict[str, Any],
        vocabulary: Dict[str, Any],
        top_problems: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    badges = []

    new_items = vocabulary.get("new_candidates", []) or []
    activated = vocabulary.get("activated", []) or []
    uncertain = vocabulary.get("uncertain_top20", []) or []
    russian = vocabulary.get("russian_used", []) or []

    badges.append({
        "type": "new_items_used",
        "label": f"🧩 {len(new_items)} новых слова/фразы попробовал использовать",
        "items": [
            {
                "item": r["item"],
                "difficulty_level_cefr": r.get("difficulty_level_cefr"),
                "student_used_count": r.get("student_used_count"),
                "example": r.get("example"),
            }
            for r in new_items[:6]
            if r.get("example")
        ],
    })

    badges.append({
        "type": "activated_items",
        "label": f"✅ {len(activated)} слова использовал уверенно",
        "items": [
            {
                "item": r["item"],
                "difficulty_level_cefr": r.get("difficulty_level_cefr"),
                "mastery_delta": r.get("mastery_delta"),
                "example": (r.get("examples") or [None])[0],
            }
            for r in activated[:6]
            if (r.get("examples") or [])
        ],
    })

    badges.append({
        "type": "top_problems",
        "label": "⚠️ Основные трудности урока",
        "items": top_problems[:2],
    })

    ratio = float(talk_metrics.get("student_speaking_ratio", 0.0) or 0.0)
    badges.append({
        "type": "student_centered",
        "label": f"🎯 Сегодня ты говорил {int(round(ratio * 100))}% времени",
        "details": {
            "student_time_sec": int(talk_metrics.get("student_speaking_time_sec") or 0),
            "teacher_time_sec": int(talk_metrics.get("teacher_speaking_time_sec") or 0),
            "student_words": int(talk_metrics.get("student_words_count") or 0),
            "teacher_words": int(talk_metrics.get("teacher_words_count") or 0),
        }
    })

    if russian:
        badges.append({
            "type": "russian_used",
            "label": f"🇷🇺 {len(russian)} русских слов — добавить перевод",
            "items": russian[:3],
        })

    return badges[:6]


def build_lesson_report(
        *,
        schema_version: str,
        lesson_id: str,
        user_id: int,
        source: Dict[str, Any],
        talk_metrics: Dict[str, Any],
        vocabulary_block: Dict[str, Any],
        mind_map: Dict[str, Any],
        action_items: Dict[str, Any],
) -> Dict[str, Any]:
    uncertain_top20 = (vocabulary_block.get("uncertain_top20") or [])
    top_problems = build_top_problems_from_uncertain(uncertain_top20)

    return {
        "schema_version": schema_version,
        "lesson_id": lesson_id,
        "user_id": user_id,
        "created_at": now_iso(),
        "source": source,
        "talk_metrics": talk_metrics,
        "badges": build_badges(
            talk_metrics=talk_metrics,
            vocabulary=vocabulary_block,
            top_problems=top_problems,
        ),
        "vocabulary": vocabulary_block,
        "mind_map": mind_map,
        "action_items": action_items,
    }
