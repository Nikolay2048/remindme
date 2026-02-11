from __future__ import annotations

import os
import re
import time
from pathlib import Path

from aiogram import Bot
from aiogram.types import Message

SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
MAX_TG_FILE_SIZE = 20 * 1024 * 1024  # Telegram Bot API getFile download limit


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "file"
    name = SAFE_NAME_RE.sub("_", name)
    return name[:120]


async def download_telegram_file(bot: Bot, message: Message, out_dir: str) -> str:
    """
    Supports: video, audio, document (video/audio sent as document also works).
    Returns local path.
    """
    ensure_dir(out_dir)

    file = None
    original_name = None

    if message.video:
        file = message.video
        original_name = message.video.file_name or "video.mp4"
    elif message.audio:
        file = message.audio
        original_name = message.audio.file_name or "audio.mp3"
    elif message.document:
        file = message.document
        original_name = message.document.file_name or "document.bin"

    if file is None:
        raise ValueError("No supported media in message")
    if file.file_size and file.file_size > MAX_TG_FILE_SIZE:
        raise ValueError("FILE_TOO_LARGE")

    base = sanitize_filename(original_name)
    ts = int(time.time())
    local_path = os.path.join(out_dir, f"{ts}_{base}")

    tg_file = await bot.get_file(file.file_id)
    await bot.download_file(tg_file.file_path, destination=local_path, timeout=600)

    return local_path
