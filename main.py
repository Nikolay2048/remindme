import datetime
import json

from dotenv import load_dotenv

from src.features import process_video_without_captions
from src.models.user_data import UserData, UserDataYandexTranslator, UserDataTikTok
from src.services.tik_tok_processor import TikTokProcessor
# from src.services.tik_tok_processor import TikTokProcessor
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

if __name__ == '__main__':
    user_data = UserData(
        id=USER_ID,
        user_name=USER_NAME,
        email="",
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
    logger.info(f"Collected all captions for user")
    user_data_service.process_user_data()
    process_video_without_captions()

    # yandex_translator_processor = YandexTranslatorProcessor(user_data=user_data.yandex_translator)
    # yandex_translator_processor.collect_words_from_collections()
