from dotenv import load_dotenv

load_dotenv()

import logging.config
import os

from config.logging_config import LOGGING_CONFIG
from src.dataproviders.tik_tok import process_user_data_info
from src.features import process_video_without_captions

logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)

USER_DATA_PATH_ARCH = os.getenv("INPUT_USER_INFO_TIKTOK_FILE")

if __name__ == '__main__':
    logger.info(f"Starting collect user history from archive {USER_DATA_PATH_ARCH}")
    process_user_data_info(USER_DATA_PATH_ARCH)
    process_video_without_captions()
