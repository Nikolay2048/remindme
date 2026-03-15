from __future__ import annotations

from aiogram.fsm.state import State, StatesGroup


class ProfileFlow(StatesGroup):
    waiting_upload_choice = State()
    waiting_level = State()
    waiting_goal = State()


class UploadLectureState(StatesGroup):
    waiting_student_audio = State()
    waiting_teacher_audio = State()