from __future__ import annotations

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


class KB:
    MENU = "menu"
    CANCEL = "cancel"

    # main menu
    UPLOAD_DATA = "upload_data"
    PROGRESS = "progress"
    GET_TASK = "get_task"
    CHAT = "chat"
    ABOUT = "about"

    # upload data
    UPLOAD_LESSON = "upload_lesson"
    UPLOAD_YANDEX_TRANSLATOR_LINK = "upload_yandex_translator"


def main_menu() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="Загрузить данные", callback_data=KB.UPLOAD_DATA)
    b.button(text="Чат с AI", callback_data=KB.CHAT)
    b.button(text="Получить задание", callback_data=KB.GET_TASK)
    b.button(text="Мой прогресс", callback_data=KB.PROGRESS)
    b.button(text="About", callback_data=KB.ABOUT)
    b.adjust(1)
    return b.as_markup()


def back() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="Назад", callback_data=KB.MENU)
    b.adjust(1)
    return b.as_markup()

def upload_data() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="Загрузить урок", callback_data=KB.UPLOAD_LESSON)
    b.button(text="Загрузить ссылку на подборку яндекс переводчика", callback_data=KB.UPLOAD_YANDEX_TRANSLATOR_LINK)
    b.button(text="Назад", callback_data=KB.MENU)
    b.adjust(1)
    return b.as_markup()


# def levels_menu() -> InlineKeyboardMarkup:
#     b = InlineKeyboardBuilder()
#     b.button(text="A1", callback_data=KB.LEVEL_A1)
#     b.button(text="A2", callback_data=KB.LEVEL_A2)
#     b.button(text="B1", callback_data=KB.LEVEL_B1)
#     b.button(text="B2", callback_data=KB.LEVEL_B2)
#     b.button(text="Отмена", callback_data=KB.CANCEL)
#     b.adjust(2, 2, 1)
#     return b.as_markup()
