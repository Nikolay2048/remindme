from __future__ import annotations

import asyncio
import os
from pathlib import Path

from aiogram import F, Router
from aiogram.enums import ContentType
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from .keyboards import *
from .states import ProfileFlow
from .texts import Txt
from ...services.lesson_analyzer.processor import LessonProcessor

router = Router()
BASE_DIR = Path(__file__).resolve().parents[3]
DEVICE = os.getenv("DEVICE")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE")
UPLOAD_DIR = Path(os.getenv("LESSON_OUT_DIR"))
UPLOAD_DIR.mkdir(exist_ok=True)


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(Txt.START, reply_markup=main_menu())


@router.callback_query(F.data == KB.MENU)
async def cb_menu(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.message.edit_text(Txt.MENU, reply_markup=main_menu())
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


@router.callback_query(F.data == KB.UPLOAD_LESSON)
async def upload_lesson(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.message.edit_text(Txt.UPLOAD_LESSON, reply_markup=back())
    await state.clear()


@router.message(F.content_type == ContentType.AUDIO)
async def handle_files(message: Message):
    wait_msg = await message.answer("Файл получен. Начинаю обработку...")
    asyncio.create_task(process_audio_background(message, wait_msg.message_id))

async def process_audio_background(message: Message, wait_message_id: int):
    try:
        doc = message.audio
        file_name = doc.file_name or "file.bin"
        user_id = message.from_user.id
        username = message.from_user.username  # это @nickname, может быть None
        full_name = message.from_user.full_name
        print(user_id, full_name, username)

        if "teacher" not in file_name.lower() and "student" not in file_name.lower():
            await message.edit_text("Название файла должно содержать 'teacher' или 'student'.", reply_markup=back())
            return

        tg_file = await message.bot.get_file(doc.file_id)

        save_path = UPLOAD_DIR / file_name
        await message.bot.download_file(tg_file.file_path, destination=save_path)

        analyzer = LessonProcessor(whisper_model_size="large", device=DEVICE, compute_type=COMPUTE_TYPE)


        asyncio.create_task(analyzer.process_video_to_db(
            # user_id="user_data.id",
            user_id=user_id,
            video_path=str(save_path),
            out_dir=str(BASE_DIR / "data" / "processed" / "lessons"),
            title="Lesson NNN",
            source="file",
            materials_text=None,
        ))


        await message.bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=wait_message_id,
            text="Обработка завершена. Файл готов."
        )

    except Exception:
        await message.bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=wait_message_id,
            text="Ошибка при обработке файла. Нажмите «Назад» и попробуйте снова."
        )

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


@router.message(F.text)
async def echo_text(message: Message) -> None:
    await message.answer(
        "Я получил сообщение.\n"
        "Для меню: /menu\n"
        "Для отмены текущего шага: /cancel"
    )
