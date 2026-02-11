from __future__ import annotations

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

from .handlers import (
    Flow,
    cmd_start,
    cb_main,
    cb_how,
    cb_upload,
    cb_upload_video,
    cb_upload_audio,
    cb_upload_text,
    handle_media,
    handle_text_materials,
    cb_report,
    cb_new_words,
    cb_uncertain,
    cb_plan,
    cb_mindmap,
    cb_vocab,
    cb_vocab_category,
    cb_progress,
)
from .keyboards import KB


async def run_telegram_bot():
    bot = Bot(token=os.getenv("BOT_TOKEN"))
    dp = Dispatcher(storage=MemoryStorage())

    # /start
    dp.message.register(cmd_start, CommandStart())

    # callbacks
    dp.callback_query.register(cb_main, F.data == KB.BACK)
    dp.callback_query.register(cb_how, F.data == KB.HOW)
    dp.callback_query.register(cb_upload, F.data == KB.UPLOAD)

    dp.callback_query.register(cb_upload_video, F.data == KB.UPLOAD_VIDEO)
    dp.callback_query.register(cb_upload_audio, F.data == KB.UPLOAD_AUDIO)
    dp.callback_query.register(cb_upload_text, F.data == KB.UPLOAD_TEXT)

    dp.callback_query.register(cb_report, F.data == KB.REPORT)
    dp.callback_query.register(cb_new_words, F.data == KB.NEW_WORDS)
    dp.callback_query.register(cb_uncertain, F.data == KB.UNCERTAIN)
    dp.callback_query.register(cb_plan, F.data == KB.PLAN)
    dp.callback_query.register(cb_mindmap, F.data == KB.MINDMAP)

    dp.callback_query.register(cb_vocab, F.data == KB.VOCAB)
    dp.callback_query.register(cb_progress, F.data == KB.PROGRESS)

    dp.callback_query.register(lambda c: cb_vocab_category(c, status="new"), F.data == KB.VOCAB_NEW)
    dp.callback_query.register(lambda c: cb_vocab_category(c, status="learning"), F.data == KB.VOCAB_LEARNING)
    dp.callback_query.register(lambda c: cb_vocab_category(c, status="known"), F.data == KB.VOCAB_KNOWN)
    dp.callback_query.register(lambda c: cb_vocab_category(c, item_type="RUSSIAN_WORD"), F.data == KB.VOCAB_RU)

    # media uploads
    lesson_out_dir = os.getenv("LESSON_OUT_DIR")

    async def _handle_text_materials(message, state):
        await handle_text_materials(message, state, lesson_out_dir)

    async def _handle_media(message, state, bot):
        await handle_media(message, state, bot, lesson_out_dir)

    dp.message.register(_handle_text_materials, Flow.waiting_text, F.text)
    dp.message.register(_handle_media, Flow.waiting_video_or_audio, (F.video | F.audio | F.document))

    await dp.start_polling(bot)


