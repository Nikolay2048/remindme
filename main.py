import datetime
import json
from pathlib import Path

from dotenv import load_dotenv

from src.models.user_data import UserData, UserDataYandexTranslator, UserDataTikTok
from src.services.lesson_analyzer.processor import LessonProcessor
from src.services.tik_tok_processor import TikTokProcessor
from src.services.user_data_processor import UserDataService

load_dotenv()

import logging.config
import os

from config.logging_config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)

VIDEO_WORKER_COUNT = int(os.getenv("VIDEO_API_PARSER_WORKER_COUNT"))
VIDEO_BATCH_SIZE = int(os.getenv("VIDEO_DB_BATCH_SIZE"))

USER_DATA_TIK_TOK_PATH_ARCH = os.getenv("INPUT_USER_INFO_TIKTOK_FILE")
USER_YANDEX_TRANSLATOR_COLLECTIONS = os.getenv("USER_YANDEX_TRANSLATOR_COLLECTIONS")

# from tg\ui
USER_ID = 123
USER_NAME = "user"

BASE_DIR = Path(__file__).resolve().parent
video_path = BASE_DIR / "data" / "lessons" / "Arina_lesson_2.mp4"

if __name__ == '__main__':
    user_data = UserData(
        id=USER_ID,
        tg_username="tg_id",
        user_name=USER_NAME,
        email="@emal",
        created_at=datetime.datetime.now(),
        last_visit_at=datetime.datetime.now(),
        tiktok_data=UserDataTikTok(USER_DATA_TIK_TOK_PATH_ARCH),
        yandex_translator=UserDataYandexTranslator(
            collections=json.loads(USER_YANDEX_TRANSLATOR_COLLECTIONS))
    )
    logger.info(f"Starting collect user history from archive {USER_DATA_TIK_TOK_PATH_ARCH}")

    user_data_service = UserDataService(user_data=user_data)
    tik_tok_processor = TikTokProcessor(video_links_list=user_data_service.get_user_video_links())
    tik_tok_processor.collect_video_captions_from_user_videos(VIDEO_BATCH_SIZE, VIDEO_WORKER_COUNT)
    logger.info(f"Collected all captions from tik-tok for user")
    # process_video_without_captions()
    #
    logger.info(f"Processing user data")
    user_data_service.process_user_data()

    analyzer = LessonProcessor(whisper_model_size="large")
    res = analyzer.process_video_to_db(
        user_id=USER_ID,
        video_path=str(video_path),
        out_dir=str(BASE_DIR / "data" / "processed" / "lessons"),
        title="Lesson 2",
        source="file",
        materials_text=None,
    )
    print("OK lesson:", res["lesson_db_id"])
    print("Report:", res["paths"]["report_json"])
