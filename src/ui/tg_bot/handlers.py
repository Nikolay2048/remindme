from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from uuid import uuid4

from aiogram import F, Router
from aiogram.enums import ContentType
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message, ReplyKeyboardRemove

from .handler_utils import save_audio_file, process_lesson_background
from .keyboards import *
from .states import ProfileFlow, UploadLectureState, UploadTranslatorLinksState, ChatAIState
from .texts import Txt
from ...services.lesson_analyzer.processor import LessonProcessor
from ...services.translator_processor import YandexTranslatorProcessor

logger = logging.getLogger(__name__)

router = Router()
BASE_DIR = Path(__file__).resolve().parents[3]
DEVICE = os.getenv("DEVICE")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE")


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(Txt.START, reply_markup=main_menu())


@router.message(Command("menu"))
@router.message(F.text == KB.MENU)
async def cb_menu(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        Txt.MENU,
        reply_markup=main_menu()
    )

@router.callback_query(F.data == KB.MENU)
async def cb_menu_callback(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.message.edit_text(
        Txt.MENU,
        reply_markup=main_menu()
    )
    await callback.answer()

@router.callback_query(F.data == KB.ABOUT)
async def cb_about(callback: CallbackQuery) -> None:
    await callback.message.edit_text(Txt.ABOUT, reply_markup=back())
    await callback.answer()


@router.callback_query(F.data == KB.UPLOAD_DATA)
async def user_upload_data(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(ProfileFlow.waiting_upload_choice)
    await callback.message.edit_text(Txt.UPLOAD_CHOOSE, reply_markup=upload_data())
    await callback.answer()


@router.callback_query(F.data == KB.GET_TASK)
async def get_user_task(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()

    await callback.message.edit_text(Txt.UNIMPLEMENTED, reply_markup=back())
    await callback.answer()


@router.callback_query(F.data == KB.PROGRESS)
async def get_progress(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()

    await callback.message.edit_text(Txt.UNIMPLEMENTED, reply_markup=back())
    await callback.answer()


@router.callback_query(F.data == KB.UPLOAD_YANDEX_TRANSLATOR_LINK)
async def upload_translator_link(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()

    await state.set_state(UploadTranslatorLinksState.waiting_links)
    await state.update_data(translator_links=[])

    await callback.message.edit_text(Txt.UPLOAD_TRANSLATOR_LINK, reply_markup=back())
    await callback.answer()


@router.message(UploadTranslatorLinksState.waiting_links, F.text)
async def handle_translator_link(message: Message, state: FSMContext) -> None:
    url = (message.text or "").strip()

    data = await state.get_data()
    links = data.get("translator_links", [])
    links.append(url)
    await state.update_data(translator_links=links)

    await message.answer("⏳ Ссылка получена. Начинаю обработку...")

    try:
        translator_processor = YandexTranslatorProcessor()
        result = translator_processor.process_user_collections(
            message.from_user.id,
            [url]
        )

        if asyncio.iscoroutine(result):
            await result

        await message.answer(
            f"Ссылка обработана.\n"
            f"Всего ссылок в текущей сессии: {len(links)}",
            reply_markup=back()
        )

    except Exception as e:
        logger.exception("Ошибка при обработке ссылки переводчика")
        await message.answer(
            "Не удалось обработать ссылку. Проверьте корректность введенных данных",
            reply_markup=back()
        )


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


@router.message(ChatAIState.chatting, F.text == KB.EXIT_CHAT)
async def exit_ai_chat(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "Вы вышли из режима чата.",
        reply_markup=ReplyKeyboardRemove()
    )
    await message.answer(Txt.MENU, reply_markup=main_menu())


async def ask_ai(text: str) -> str:
    response = "response AI"
    return response


@router.message(ChatAIState.chatting, F.text)
async def ai_chat(message: Message, state: FSMContext):
    user_text = message.text

    try:
        ai_reply = await ask_ai(user_text)

        await message.answer(ai_reply)

    except Exception:
        logger.exception("Ошибка AI чата")

        await message.answer(
            "⚠️ Ошибка при обращении к AI."
        )


@router.callback_query(F.data == KB.CHAT)
async def start_ai_chat(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(ChatAIState.chatting)

    await callback.message.edit_text(
        "💬 Вы вошли в режим чата с AI.\n\n"
        "Напишите сообщение.\n"
        "Чтобы выйти, нажмите «⬅️ Выйти из чата» или отправьте /menu.",
        reply_markup=back()
    )

    await callback.message.answer(
        "Режим чата активирован.",
        reply_markup=chat_keyboard()
    )

    await callback.answer()


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
