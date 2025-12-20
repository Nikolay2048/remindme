import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path

from aiogram.enums import ParseMode
from aiogram.types import Message

from src.services.lesson_analyzer.processor import LessonProcessor
from src.ui.tg_bot.keyboards import back
from src.ui.tg_bot.texts import Txt

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.getenv("LESSON_OUT_DIR"))
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_LEN = 4096


async def save_audio_file(message: Message, role: str, lecture_id: str) -> Path:
    audio = message.audio
    if not audio:
        raise ValueError("Сообщение не содержит audio")

    user_id = message.from_user.id
    tg_file = await message.bot.get_file(audio.file_id)

    original_name = audio.file_name or f"{role}.mp3"
    suffix = Path(original_name).suffix or ".mp3"
    safe_name = f"{role}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"

    lecture_dir = UPLOAD_DIR / str(user_id) / lecture_id
    lecture_dir.mkdir(parents=True, exist_ok=True)

    save_path = lecture_dir / safe_name
    await message.bot.download_file(tg_file.file_path, destination=save_path, timeout=120)

    return save_path


async def process_lesson_background(
        *,
        bot,
        chat_id: int,
        wait_message_id: int,
        analyzer: LessonProcessor,
        user_id: int,
        lesson_id: str,
        student_audio_path: Path,
        teacher_audio_path: Path,
):
    try:
        result = await analyzer.process_lesson_to_db(
            user_id=user_id,
            lesson_id=lesson_id,
            student_audio_path=str(student_audio_path),
            teacher_audio_path=str(teacher_audio_path),
            lesson_dir=str(student_audio_path.parent),
            title="Lesson",
            source="uploaded_tg_file",
            materials_text=None,
        )

        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=wait_message_id,
            text="✅ Обработка лекции успешно завершена.",
            reply_markup=back(),
        )

        await send_lesson_report(bot, chat_id, result["paths"]["report_json"])

    except Exception:
        logger.error(traceback.format_exc())

        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=wait_message_id,
            text="❌ Ошибка при обработке лекции. Попробуйте загрузить файлы снова.",
            reply_markup=back(),
        )


def split_text_by_blocks(blocks: list[str], prefix: str = "", max_len: int = MAX_LEN) -> list[str]:
    """
    Собирает сообщения из готовых блоков так, чтобы каждый блок целиком
    помещался в сообщение. prefix добавляется только к первому сообщению.
    """
    messages = []
    current = prefix

    for block in blocks:
        # Если один блок сам по себе слишком большой — режем его грубо
        if len(block) > max_len:
            if current:
                messages.append(current)
                current = ""

            for i in range(0, len(block), max_len):
                part = block[i:i + max_len]
                messages.append(part)
            continue

        if len(current) + len(block) > max_len:
            if current:
                messages.append(current)
            current = block
        else:
            current += block

    if current:
        messages.append(current)

    return messages


async def send_long_message(bot, chat_id: int, text: str, parse_mode: ParseMode = ParseMode.HTML):
    """
    Отправляет длинный текст частями.
    Для простого текста без гарантии сохранения HTML-структуры между чанками.
    """
    for i in range(0, len(text), MAX_LEN):
        await bot.send_message(
            chat_id=chat_id,
            text=text[i:i + MAX_LEN],
            parse_mode=parse_mode
        )


async def send_report_section(bot, chat_id: int, template: str, items: list[str], key: str):
    """
    Безопасно собирает section-отчёт по блокам.
    template должен быть вида:
      "... {new_items_used}"
      "... {activated_items}"
    """
    empty_text = template.format(**{key: ""})

    chunks = split_text_by_blocks(
        blocks=items,
        prefix=empty_text,
        max_len=MAX_LEN
    )

    for chunk in chunks:
        await bot.send_message(
            chat_id=chat_id,
            text=chunk,
            parse_mode=ParseMode.HTML
        )


async def send_lesson_report(bot, chat_id, report_path: str):
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    vocabulary = report.get("vocabulary", {})
    talk_metrics = report.get("talk_metrics", {})

    new_items_blocks = []
    for w in vocabulary.get("new_candidates", []):
        if w.get("example"):
            block = Txt.WORD_REPORT.format(
                word=w.get("item", ""),
                count=w.get("student_used_count", ""),
                difficulty_level_cefr=w.get("difficulty_level_cefr", ""),
                context=w.get("example", {}).get("context", "")
            )
            new_items_blocks.append(block)

    activated_blocks = []
    for w in vocabulary.get("activated", []):
        if w.get("examples"):
            block = Txt.WORD_REPORT.format(
                word=w.get("item", ""),
                count=w.get("student_used_count", ""),
                difficulty_level_cefr=w.get("difficulty_level_cefr", ""),
                context=", ".join(ex.get("context", "") for ex in w.get("examples", []))
            )
            activated_blocks.append(block)

    russian_used_blocks = []
    for w in vocabulary.get("russian_used", []):
        if w.get("example"):
            block = Txt.WORD_REPORT.format(
                word=w.get("item", ""),
                count=w.get("student_used_count", ""),
                difficulty_level_cefr=w.get("difficulty_level_cefr", ""),
                context=w.get("example", {}).get("context", "")
            )
            russian_used_blocks.append(block)

    await bot.send_message(
        chat_id=chat_id,
        text=Txt.LESSON_REPORT[0].format(
            datetime=report.get("created_at", ""),
            student_speaking_time_min=int(talk_metrics.get("student_speaking_time_sec", 0)) / 60,
            teacher_speaking_time_min=int(talk_metrics.get("teacher_speaking_time_sec", 0)) / 60,
            student_words_count=talk_metrics.get("student_words_count", 0),
            teacher_words_count=talk_metrics.get("teacher_words_count", 0),
        ),
        parse_mode=ParseMode.HTML
    )

    await send_report_section(
        bot=bot,
        chat_id=chat_id,
        template=Txt.LESSON_REPORT[1],
        items=new_items_blocks,
        key="new_items_used"
    )

    await send_report_section(
        bot=bot,
        chat_id=chat_id,
        template=Txt.LESSON_REPORT[2],
        items=activated_blocks,
        key="activated_items"
    )

    await send_report_section(
        bot=bot,
        chat_id=chat_id,
        template=Txt.LESSON_REPORT[3],
        items=russian_used_blocks,
        key="russian_used"
    )
