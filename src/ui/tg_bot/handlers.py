from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from aiogram import F, Router
from aiogram.enums import ContentType, ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from spacy.lang.en.tokenizer_exceptions import word

from .keyboards import *
from .states import ProfileFlow, UploadLectureState
from .texts import Txt
from ...services.lesson_analyzer.processor import LessonProcessor

logger = logging.getLogger(__name__)

router = Router()
BASE_DIR = Path(__file__).resolve().parents[3]
DEVICE = os.getenv("DEVICE")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE")
UPLOAD_DIR = Path(os.getenv("LESSON_OUT_DIR"))
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_LEN = 4096

@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(Txt.START, reply_markup=main_menu())


@router.callback_query(F.data == KB.MENU)
async def cb_menu(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.message.edit_text(Txt.MENU, reply_markup=main_menu())
    await callback.answer()


@router.message(Command("menu"))
async def cb_menu(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(Txt.MENU, reply_markup=main_menu())


@router.callback_query(F.data == KB.ABOUT)
async def cb_about(callback: CallbackQuery) -> None:
    await callback.message.edit_text(Txt.ABOUT, reply_markup=back())
    await callback.answer()


@router.callback_query(F.data == KB.UPLOAD_DATA)
async def user_upload_data(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(ProfileFlow.waiting_upload_choice)
    await callback.message.edit_text(Txt.UPLOAD_CHOOSE, reply_markup=upload_data())
    await callback.answer()


@router.callback_query(F.data == KB.UPLOAD_LESSON)
async def upload_lesson(callback: CallbackQuery, state: FSMContext) -> None:
    lecture_id = str(uuid4())

    await state.clear()
    await state.set_state(UploadLectureState.waiting_student_audio)
    await state.update_data(
        lecture_id=lecture_id,
        student_audio_path=None,
        teacher_audio_path=None,
    )

    await callback.message.edit_text(
        "Загрузка урока.\n\nПришлите аудиодорожку ученика.",
        reply_markup=back(),
    )
    await callback.answer()


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


@router.message(UploadLectureState.waiting_student_audio, F.content_type == ContentType.AUDIO)
async def handle_student_audio(message: Message, state: FSMContext):
    data = await state.get_data()
    lecture_id = data["lecture_id"]

    await message.answer("Аудиофайл ученика получен. Скачиваю файл.")
    try:
        student_audio_path = await save_audio_file(
            message=message,
            role="student",
            lecture_id=lecture_id,
        )
    except Exception:
        logger.exception("Ошибка при сохранении аудио ученика")
        await message.answer("Не удалось сохранить аудио ученика.")
        return

    await state.update_data(student_audio_path=student_audio_path)
    await state.set_state(UploadLectureState.waiting_teacher_audio)

    await message.answer("Аудио ученика сохранено. Теперь пришлите аудиодорожку учителя.")


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


@router.message(UploadLectureState.waiting_teacher_audio, F.content_type == ContentType.AUDIO)
async def handle_teacher_audio(message: Message, state: FSMContext):
    data = await state.get_data()
    lecture_id = data["lecture_id"]
    student_audio_path = data.get("student_audio_path")

    if not student_audio_path:
        await state.clear()
        await message.answer("Не найден файл ученика. Начните загрузку заново.")
        return

    try:
        teacher_audio_path = await save_audio_file(
            message=message,
            role="teacher",
            lecture_id=lecture_id,
        )
    except Exception:
        logger.exception("Ошибка при сохранении аудио учителя")
        await message.answer("Не удалось сохранить аудио учителя.")
        return

    await state.update_data(teacher_audio_path=teacher_audio_path)

    wait_msg = await message.answer(
        "⏳ Обе дорожки получены. Начинаю обработку лекции..."
    )

    user_id = message.from_user.id

    analyzer = LessonProcessor(
        whisper_model_size="large",
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    asyncio.create_task(
        process_lesson_background(
            bot=message.bot,
            chat_id=message.chat.id,
            wait_message_id=wait_msg.message_id,
            analyzer=analyzer,
            user_id=user_id,
            lesson_id=lecture_id,
            student_audio_path=student_audio_path,
            teacher_audio_path=teacher_audio_path,
        )
    )

    await state.clear()


# @router.message(ProfileFlow.waiting_name, F.text)
# async def profile_name(message: Message, state: FSMContext) -> None:
#     name = (message.text or "").strip()
#     if not name:
#         await message.answer("Имя не должно быть пустым.")
#         return
#
#     await state.update_data(name=name)
#     await state.set_state(ProfileFlow.waiting_level)
#     await message.answer(Txt.ASK_LEVEL, reply_markup=levels_menu())


# @router.callback_query(ProfileFlow.waiting_level, F.data.in_(set(LEVEL_BY_CB.keys())))
# async def profile_level(callback: CallbackQuery, state: FSMContext) -> None:
#     level = LEVEL_BY_CB[callback.data]
#     await state.update_data(level=level)
#     await state.set_state(ProfileFlow.waiting_goal)
#     await callback.message.edit_text(Txt.ASK_GOAL, reply_markup=back_to_menu())
#     await callback.answer()


@router.message(UploadLectureState.waiting_student_audio)
async def wrong_student_input(message: Message):
    await message.answer("Ожидается аудиодорожка ученика.")


@router.message(UploadLectureState.waiting_teacher_audio)
async def wrong_teacher_input(message: Message):
    await message.answer("Ожидается аудиодорожка учителя.")


@router.message(F.text)
async def echo_text(message: Message) -> None:
    await message.answer(
        "Я получил сообщение. На данном шаге я не обрабатываю текстовые сообщения\n"
        "Для меню: /menu\n"
        "Для отмены текущего шага: /cancel"
    )
