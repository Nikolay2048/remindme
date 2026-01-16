from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message

from .messages import Msg
from .keyboards import KB, main_menu, upload_choose, back_to_main, lesson_ready_menu, report_menu, vocab_menu
from .utils import download_telegram_file, ensure_dir


# ====== IMPORT твоего пайплайна ======
# Проверь, что LessonProcessor реально лежит тут.
# Например: src/services/lesson_analyzer/processor.py -> LessonProcessor
from src.services.lesson_analyzer.processor import LessonProcessor

# DB репозитории (для прогресса/словаря)
from src.db.repositories import UserRepository, UserVocabularyRepository
from src.models.user_data import UserData


class Flow(StatesGroup):
    waiting_video_or_audio = State()
    waiting_text = State()


@dataclass
class LastLessonCache:
    lesson_db_id: Optional[int] = None
    report: Optional[Dict[str, Any]] = None


# простейший in-memory cache по user_id
LAST: dict[int, LastLessonCache] = {}


def _get_cache(user_id: int) -> LastLessonCache:
    if user_id not in LAST:
        LAST[user_id] = LastLessonCache()
    return LAST[user_id]


def _format_lesson_summary(report: Dict[str, Any]) -> str:
    tm = report.get("talk_metrics") or {}
    vocab = report.get("vocabulary") or {}
    agg = report.get("aggregates") or {}

    ratio = tm.get("student_speaking_ratio")
    ratio_pct = int(round((ratio or 0) * 100))

    return (
        "✅ **Урок проанализирован**\n\n"
        "Коротко по результатам:\n\n"
        f"🕒 Ты говорил **{ratio_pct}% времени**\n"
        f"🧩 Новых слов: **{agg.get('new_items_count', 0)}**\n"
        f"✅ Уверенно использованных слов: **{agg.get('activated_items_count', 0)}**\n"
        f"⚠️ Основных сложных мест: **{agg.get('uncertain_items_count', 0)}**"
    )


def _format_report_text(report: Dict[str, Any]) -> str:
    tm = report.get("talk_metrics") or {}
    vocab = report.get("vocabulary") or {}
    ratio = tm.get("student_speaking_ratio") or 0.0
    ratio_pct = int(round(ratio * 100))

    teacher_wc = tm.get("teacher_words_count", 0)
    student_wc = tm.get("student_words_count", 0)

    new_cnt = (report.get("aggregates") or {}).get("new_items_count", 0)
    act_cnt = (report.get("aggregates") or {}).get("activated_items_count", 0)

    uncertain = (vocab.get("uncertain_top20") or [])
    p1 = uncertain[0]["item"] if len(uncertain) >= 1 else "—"
    p2 = uncertain[1]["item"] if len(uncertain) >= 2 else "—"

    return (
        "📄 **Отчёт по уроку**\n\n"
        "🕒 Разговор:\n"
        f"• Ты говорил **{ratio_pct}% времени**\n"
        f"• Слов: ты — {student_wc}, преподаватель — {teacher_wc}\n\n"
        "🧩 Словарный прогресс:\n"
        f"• Новых слов/фраз: {new_cnt}\n"
        f"• Активированных слов: {act_cnt}\n\n"
        "⚠️ Основной фокус:\n"
        f"• {p1}\n"
        f"• {p2}"
    )


def _format_new_words(report: Dict[str, Any]) -> str:
    rows = ((report.get("vocabulary") or {}).get("new_candidates") or [])[:20]
    if not rows:
        return "🧩 **Новые слова и фразы**\n\nПохоже, явных новых слов не нашёл в этом уроке."

    lines = ["🧩 **Новые слова и фразы**\n\nВот что ты попробовал использовать:\n"]
    for r in rows:
        item = r.get("item")
        lvl = r.get("difficulty_level_cefr", "UNKNOWN")
        ex = r.get("example") or {}
        ctx = ex.get("context")
        if ctx:
            ctx = ctx.strip()
            if len(ctx) > 140:
                ctx = ctx[:140] + "…"
        lines.append(f"• **{item}** ({lvl})" + (f"\n  _{ctx}_" if ctx else ""))

    return "\n".join(lines)


def _format_uncertain(report: Dict[str, Any]) -> str:
    rows = ((report.get("vocabulary") or {}).get("uncertain_top20") or [])[:10]
    if not rows:
        return "⚠️ **Над чем поработать**\n\nЯвных проблемных слов не увидел — это хороший знак."

    lines = ["⚠️ **Над чем поработать**\n\nСлова, где чаще встречалась пауза/неуверенность:\n"]
    for r in rows:
        item = r.get("item")
        why = ", ".join(r.get("why") or [])
        ex = r.get("example") or {}
        ctx = ex.get("context")
        if ctx:
            ctx = ctx.strip()
            if len(ctx) > 140:
                ctx = ctx[:140] + "…"
        lines.append(f"• **{item}** ({why or 'uncertain'})" + (f"\n  _{ctx}_" if ctx else ""))

    return "\n".join(lines)


def _format_plan(report: Dict[str, Any]) -> str:
    ai = report.get("action_items") or {}
    words = ai.get("repeat_words") or []
    goal = ai.get("speaking_goal") or "Сделать ответы короче и увереннее"

    w = ", ".join(words[:5]) if words else "—"
    return (
        "🎯 **Фокус на следующий урок**\n\n"
        f"🔹 Цель:\n{goal}\n\n"
        f"🔁 Повторить слова:\n{w}"
    )


def _format_mindmap(report: Dict[str, Any]) -> str:
    mm = report.get("mind_map") or {}
    segs = mm.get("segments") or []
    if not segs:
        return "🧠 **Структура урока**\n\nПока не удалось построить структуру."

    lines = ["🧠 **Структура урока**\n\nЯ разбил урок на блоки:\n"]
    for s in segs[:6]:
        sid = s.get("segment_id")
        t0 = s.get("t_start_sec")
        t1 = s.get("t_end_sec")
        ratio = s.get("student_speaking_ratio")
        ratio_pct = int(round((ratio or 0) * 100))
        snippet = (s.get("snippet") or "").strip()
        if len(snippet) > 120:
            snippet = snippet[:120] + "…"
        lines.append(f"• #{sid} [{t0}–{t1}] — ты говорил {ratio_pct}%\n  _{snippet}_")
    return "\n".join(lines)


async def cmd_start(message: Message, state: FSMContext):
    await state.clear()

    # ensure user exists
    uid = message.from_user.id
    username = message.from_user.full_name or "user"
    tg_username = message.from_user.username

    # твоя модель UserData: подстрой под реальные поля
    try:
        UserRepository.add_user(UserData(id=uid, user_name=username, email=None))
    except Exception:
        # если уже есть — ок, не валим бота
        pass

    await message.answer(Msg.START, reply_markup=main_menu(), parse_mode="Markdown")


async def cb_main(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text(Msg.START, reply_markup=main_menu(), parse_mode="Markdown")
    await callback.answer()


async def cb_how(callback: CallbackQuery):
    await callback.message.edit_text(Msg.HOW_IT_WORKS, reply_markup=back_to_main(), parse_mode="Markdown")
    await callback.answer()


async def cb_upload(callback: CallbackQuery):
    await callback.message.edit_text(Msg.UPLOAD_CHOOSE, reply_markup=upload_choose(), parse_mode="Markdown")
    await callback.answer()


async def cb_upload_video(callback: CallbackQuery, state: FSMContext):
    await state.set_state(Flow.waiting_video_or_audio)
    await callback.message.edit_text("🎥 Ок. " + Msg.NEED_FILE, reply_markup=back_to_main(), parse_mode="Markdown")
    await callback.answer()


async def cb_upload_audio(callback: CallbackQuery, state: FSMContext):
    await state.set_state(Flow.waiting_video_or_audio)
    await callback.message.edit_text("🎧 Ок. " + Msg.NEED_FILE, reply_markup=back_to_main(), parse_mode="Markdown")
    await callback.answer()


async def cb_upload_text(callback: CallbackQuery, state: FSMContext):
    await state.set_state(Flow.waiting_text)
    await callback.message.edit_text("📝 Ок. " + Msg.NEED_TEXT, reply_markup=back_to_main(), parse_mode="Markdown")
    await callback.answer()


async def handle_text_materials(message: Message, state: FSMContext, config_out_dir: str):
    user_id = message.from_user.id
    text = (message.text or "").strip()
    if not text:
        await message.answer(Msg.NEED_TEXT)
        return

    await state.clear()
    await message.answer(Msg.UPLOAD_RECEIVED, parse_mode="Markdown")

    # в MVP: создадим "урок из текста" (без видео), сохраним только materials_text
    # если хочешь — можно позже запускать аналитику по тексту отдельно
    try:
        # Создаём запись урока + report_json минимальный
        from src.db.repositories import LessonRepository

        lesson_db_id = LessonRepository.create_lesson(
            user_id=user_id,
            title="Text lesson",
            source="text",
            source_uri=None,
            language_mode="ru+en",
            duration_sec=None,
            started_at=None,
            metadata={"pipeline": "bot_text_only"},
        )

        report = {
            "schema_version": "1.0",
            "lesson_id": str(lesson_db_id),
            "user_id": user_id,
            "source": {"type": "text"},
            "talk_metrics": {},
            "vocabulary": {},
            "mind_map": {},
            "action_items": {},
            "aggregates": {},
        }

        LessonRepository.update_lesson_asr_payload(
            lesson_db_id,
            teacher_full_text=None,
            student_full_text=None,
            dialog_utterances_jsonb=None,
            talk_metrics=None,
            report_json=report,
            materials_text=text,
        )

        cache = _get_cache(user_id)
        cache.lesson_db_id = lesson_db_id
        cache.report = report

        await message.answer(
            "✅ Текст сохранён в урок.\n\n"
            "Пока я не строю аналитику только по тексту (мы это добавим позже).",
            reply_markup=lesson_ready_menu(),
            parse_mode="Markdown"
        )

    except Exception:
        await message.answer(Msg.ERROR_GENERIC)


async def handle_media(message: Message, state: FSMContext, bot: Bot, config_out_dir: str):
    user_id = message.from_user.id

    try:
        await state.clear()
        await message.answer(Msg.UPLOAD_RECEIVED, parse_mode="Markdown")

        # сохраняем файл
        work_dir = ensure_dir(f"{config_out_dir}/tg_uploads/{user_id}")
        local_path = await download_telegram_file(bot, message, out_dir=work_dir)

        # стартуем тяжёлую обработку в отдельном потоке
        async def run_pipeline():
            processor = LessonProcessor(
                whisper_model_size="small",
                device="cuda",
                compute_type="float16",
                language="en",
            )
            return processor.process_video_to_db(
                user_id=user_id,
                video_path=local_path,
                out_dir=config_out_dir,
                title="Lesson",
                source="file",
                materials_text=None,
            )

        res = await asyncio.to_thread(lambda: asyncio.run(run_pipeline()))  # безопасно для синхронной тяжёлой части

        # загрузим report_json из файла (чтобы не зависеть от структуры res)
        report_path = res["paths"]["report_json"]
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        cache = _get_cache(user_id)
        cache.lesson_db_id = res.get("lesson_db_id")
        cache.report = report

        await message.answer(_format_lesson_summary(report), reply_markup=lesson_ready_menu(), parse_mode="Markdown")

    except ValueError:
        await message.answer(Msg.ERROR_FILE)
    except Exception:
        await message.answer(Msg.ERROR_GENERIC)


async def cb_report(callback: CallbackQuery):
    user_id = callback.from_user.id
    cache = _get_cache(user_id)
    if not cache.report:
        await callback.message.edit_text(Msg.EMPTY, reply_markup=main_menu())
        await callback.answer()
        return

    await callback.message.edit_text(_format_report_text(cache.report), reply_markup=report_menu(), parse_mode="Markdown")
    await callback.answer()


async def cb_new_words(callback: CallbackQuery):
    user_id = callback.from_user.id
    cache = _get_cache(user_id)
    if not cache.report:
        await callback.message.edit_text(Msg.EMPTY, reply_markup=main_menu())
        await callback.answer()
        return

    await callback.message.edit_text(_format_new_words(cache.report), reply_markup=back_to_main(), parse_mode="Markdown")
    await callback.answer()


async def cb_uncertain(callback: CallbackQuery):
    user_id = callback.from_user.id
    cache = _get_cache(user_id)
    if not cache.report:
        await callback.message.edit_text(Msg.EMPTY, reply_markup=main_menu())
        await callback.answer()
        return

    await callback.message.edit_text(_format_uncertain(cache.report), reply_markup=back_to_main(), parse_mode="Markdown")
    await callback.answer()


async def cb_plan(callback: CallbackQuery):
    user_id = callback.from_user.id
    cache = _get_cache(user_id)
    if not cache.report:
        await callback.message.edit_text(Msg.EMPTY, reply_markup=main_menu())
        await callback.answer()
        return

    await callback.message.edit_text(_format_plan(cache.report), reply_markup=back_to_main(), parse_mode="Markdown")
    await callback.answer()


async def cb_mindmap(callback: CallbackQuery):
    user_id = callback.from_user.id
    cache = _get_cache(user_id)
    if not cache.report:
        await callback.message.edit_text(Msg.EMPTY, reply_markup=main_menu())
        await callback.answer()
        return

    await callback.message.edit_text(_format_mindmap(cache.report), reply_markup=back_to_main(), parse_mode="Markdown")
    await callback.answer()


async def cb_vocab(callback: CallbackQuery):
    await callback.message.edit_text("📘 **Твой словарь**\n\nВыбери категорию:", reply_markup=vocab_menu(), parse_mode="Markdown")
    await callback.answer()


def _format_vocab_rows(rows: list[dict], title: str) -> str:
    if not rows:
        return f"📘 **{title}**\n\nПока пусто."

    lines = [f"📘 **{title}**\n"]
    for r in rows[:30]:
        text = r.get("text")
        lvl = r.get("difficulty_level_cefr", "UNKNOWN")
        status = r.get("status", "new")
        mastery = r.get("mastery", 0)
        lines.append(f"• **{text}** ({lvl}) — {status}, mastery={mastery}")
    return "\n".join(lines)


async def cb_vocab_category(callback: CallbackQuery, status: Optional[str] = None, item_type: Optional[str] = None):
    user_id = callback.from_user.id

    # MVP: берём всё из БД и фильтруем в памяти (потом сделаем SQL фильтры)
    vocab = UserVocabularyRepository.load_user_vocab(user_id)
    rows = list(vocab.values())

    if status:
        rows = [r for r in rows if (r.get("status") == status)]
    if item_type:
        rows = [r for r in rows if (r.get("item_type") == item_type)]

    title = "Словарь"
    if status == "new":
        title = "Новые"
    elif status == "learning":
        title = "В изучении"
    elif status == "known":
        title = "Известные"
    if item_type == "RUSSIAN_WORD":
        title = "Сказал по-русски"

    await callback.message.edit_text(_format_vocab_rows(rows, title), reply_markup=vocab_menu(), parse_mode="Markdown")
    await callback.answer()


async def cb_progress(callback: CallbackQuery):
    user_id = callback.from_user.id

    vocab = UserVocabularyRepository.load_user_vocab(user_id)
    rows = list(vocab.values())
    total = len(rows)
    active = sum(1 for r in rows if (r.get("active_knowledge") is True))

    # lessons_count быстро в MVP не считаем (нет репо метода) — покажем N/A
    text = (
        "📊 **Твой прогресс**\n\n"
        f"• Активный словарь: **{active} слов**\n"
        f"• Всего слов в базе: {total}\n"
        f"• Проанализировано уроков: N/A\n\n"
        "Ты двигаешься в правильном направлении 👍"
    )
    await callback.message.edit_text(text, reply_markup=back_to_main(), parse_mode="Markdown")
    await callback.answer()
