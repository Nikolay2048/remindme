from __future__ import annotations

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


class KB:
    # callback data
    UPLOAD = "upload"
    PROGRESS = "progress"
    VOCAB = "vocab"
    HOW = "how"

    UPLOAD_VIDEO = "upload_video"
    UPLOAD_AUDIO = "upload_audio"
    UPLOAD_TEXT = "upload_text"

    REPORT = "report"
    NEW_WORDS = "new_words"
    UNCERTAIN = "uncertain"
    PLAN = "plan"
    MINDMAP = "mindmap"

    VOCAB_NEW = "vocab_new"
    VOCAB_LEARNING = "vocab_learning"
    VOCAB_KNOWN = "vocab_known"
    VOCAB_RU = "vocab_ru"

    BACK = "back"


def main_menu() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="📤 Загрузить урок", callback_data=KB.UPLOAD)
    b.button(text="📊 Мой прогресс", callback_data=KB.PROGRESS)
    b.button(text="📘 Мой словарь", callback_data=KB.VOCAB)
    b.button(text="ℹ️ Как это работает", callback_data=KB.HOW)
    b.adjust(1)
    return b.as_markup()


def back_to_main() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="⬅️ Назад", callback_data=KB.BACK)
    return b.as_markup()


def upload_choose() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="🎥 Видео урока", callback_data=KB.UPLOAD_VIDEO)
    b.button(text="🎧 Аудио урока", callback_data=KB.UPLOAD_AUDIO)
    b.button(text="📝 Текст занятия", callback_data=KB.UPLOAD_TEXT)
    b.button(text="⬅️ Назад", callback_data=KB.BACK)
    b.adjust(1)
    return b.as_markup()


def lesson_ready_menu() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="📄 Отчёт по уроку", callback_data=KB.REPORT)
    b.button(text="🧩 Новые слова", callback_data=KB.NEW_WORDS)
    b.button(text="⚠️ Над чем поработать", callback_data=KB.UNCERTAIN)
    b.button(text="🎯 План на следующий урок", callback_data=KB.PLAN)
    b.adjust(1)
    return b.as_markup()


def report_menu() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="🧠 Mind Map урока", callback_data=KB.MINDMAP)
    b.button(text="📘 Обновлённый словарь", callback_data=KB.VOCAB)
    b.button(text="⬅️ Назад", callback_data=KB.BACK)
    b.adjust(1)
    return b.as_markup()


def vocab_menu() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="🆕 Новые", callback_data=KB.VOCAB_NEW)
    b.button(text="📈 В изучении", callback_data=KB.VOCAB_LEARNING)
    b.button(text="✅ Известные", callback_data=KB.VOCAB_KNOWN)
    b.button(text="🇷🇺 Сказал по-русски", callback_data=KB.VOCAB_RU)
    b.button(text="⬅️ Назад", callback_data=KB.BACK)
    b.adjust(1)
    return b.as_markup()
