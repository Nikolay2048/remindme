import asyncio
import datetime
import json
from pathlib import Path

from dotenv import load_dotenv

from src.db.repositories import UserRepository
from src.models.user_data import UserData
from src.services.lesson_analyzer.processor import LessonProcessor
from src.services.translator_processor import YandexTranslatorProcessor
from src.ui.tg_bot import run_bot

load_dotenv()

import logging.config
import os

from src.utils.logging_config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)

VIDEO_WORKER_COUNT = int(os.getenv("VIDEO_API_PARSER_WORKER_COUNT"))
VIDEO_BATCH_SIZE = int(os.getenv("VIDEO_DB_BATCH_SIZE"))

USER_DATA_TIK_TOK_PATH_ARCH = os.getenv("INPUT_USER_INFO_TIKTOK_FILE")
USER_YANDEX_TRANSLATOR_COLLECTIONS = os.getenv("USER_YANDEX_TRANSLATOR_COLLECTIONS")

# from tg\ui
USER_ID = 123
USER_NAME = "user"
TG_USERNAME = "tg_id"

DEVICE = os.getenv("DEVICE")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE")

BASE_DIR = Path(__file__).resolve().parent
video_path = BASE_DIR / "data" / "lessons" / "2026-01-16 18-31-17.mp4"

if __name__ == '__main__':
    logger.info(f"Start application")
    asyncio.run(run_bot())

    # user_data = UserData(
    #     id=USER_ID,
    #     tg_username=TG_USERNAME,
    #     user_name=USER_NAME,
    #     email="@emal",
    #     created_at=datetime.datetime.now(),
    #     last_visit_at=datetime.datetime.now()
    # )
    #
    # logger.info(f"Processing for user id={user_data.id}, email={user_data.email}, "
    #             f"name={user_data.user_name}")
    #
    # UserRepository.add_user(user_data)

    # ======= Tik-Tok Processing =======
    # user_data_tik_tok = UserDataTikTok(USER_DATA_TIK_TOK_PATH_ARCH)

    # tik_tok_processor = TikTokProcessor(user_data_tik_tok=user_data_tik_tok)
    # tik_tok_processor.collect_video_captions_from_user_videos(VIDEO_BATCH_SIZE, VIDEO_WORKER_COUNT,
    #                                                           user_data_tik_tok.history_video_list)
    # tik_tok_processor.update_user_video_history(user_data.id, user_data_tik_tok)

    # logger.info(f"Collected all captions from tik-tok for user")
    # process_video_without_captions()

    # translator_processor = YandexTranslatorProcessor()
    # translator_processor.process_user_collections(user_data.id, json.loads(USER_YANDEX_TRANSLATOR_COLLECTIONS))
    #
    # analyzer = LessonProcessor(whisper_model_size="large", device=DEVICE, compute_type=COMPUTE_TYPE)
    # res = analyzer.process_video_to_db(
    #     user_id=user_data.id,
    #     video_path=str(video_path),
    #     out_dir=str(BASE_DIR / "data" / "processed" / "lessons"),
    #     title="Lesson 2",
    #     source="file",
    #     materials_text=None,
    # )
    #
    # print("OK lesson:", res["lesson_db_id"])
    # print("Report:", res["paths"]["report_json"])
