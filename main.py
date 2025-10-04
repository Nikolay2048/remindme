import json

from dotenv import load_dotenv

from src.dataproviders.tik_tok_processor import TikTokProcessor
from src.dataproviders.translator_processor import YandexTranslatorProcessor
from src.features import process_video_without_captions
from src.models.user_data import UserData, UserDataYandexTranslator, UserDataTikTok

load_dotenv()

import logging.config
import os

from config.logging_config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)

VIDEO_WORKER_COUNT = int(os.getenv("VIDEO_API_PARSER_WORKER_COUNT"))
VIDEO_BATCH_SIZE = int(os.getenv("VIDEO_DB_BATCH_SIZE"))

USER_DATA_TIK_TOK_PATH_ARCH = os.getenv("INPUT_USER_INFO_TIKTOK_FILE")

if __name__ == '__main__':
    user_data = UserData(
        tiktok_data=UserDataTikTok(USER_DATA_TIK_TOK_PATH_ARCH),
        yandex_translator=UserDataYandexTranslator(
            collections=json.loads(os.getenv("USER_YANDEX_TRANSLATOR_COLLECTIONS")))
    )

    # tik_tok_processor = TikTokProcessor(user_data=user_data.tiktok_data)
    # logger.info(f"Starting collect user history from archive {USER_DATA_TIK_TOK_PATH_ARCH}")
    # tik_tok_processor.collect_video_from_user_metadata_parallel(VIDEO_BATCH_SIZE, VIDEO_WORKER_COUNT)
    # process_video_without_captions()

    yandex_translator_processor = YandexTranslatorProcessor(user_data=user_data.yandex_translator)
    yandex_translator_processor.collect_words_from_collections()
