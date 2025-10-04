import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import pyktok as pyk
import requests
from requests import RequestException

from src.data.to_postgres import insert_videos_to_database
from src.models.enums import ReasonsForSkipProcessing
from src.models.user_data import UserDataTikTok
from src.models.video import Video
from src.utils import extract_text_from_vtt
from src.utils.tik_tok_helper import get_video_id_from_video_link

logger = logging.getLogger(__name__)


class TikTokProcessor:

    def __init__(self, user_data: UserDataTikTok):
        self.user_data = user_data

    def get_caption_infos(self, video_metadata: dict) -> list | None:
        caption_infos = (
            video_metadata.get('__DEFAULT_SCOPE__', {})
            .get('webapp.video-detail', {})
            .get('itemInfo', {})
            .get('itemStruct', {})
            .get('video', {})
            .get('claInfo', {})
            .get('captionInfos', None)
        )
        return caption_infos

    def extract_caption_from_infos(self, caption_infos: list):
        if caption_infos:
            for info in caption_infos:
                for url in info.get('urlList', []):
                    try:
                        response = requests.get(url, timeout=15)
                        response.raise_for_status()
                        return extract_text_from_vtt(response.text)
                    except RequestException as e:
                        logger.debug(f"Failed to download caption from {url}: {e}")
        return None

    def get_video_metadata_from_link(self, link: str) -> Any | None:
        try:
            video_metadata = pyk.alt_get_tiktok_json(link)
            return video_metadata
        except Exception as e:
            logger.warning(f"pyktok external error. Failed to get video metadata from {link}: {e}")
            return None

    def process_video(self, video: dict, liked_links: set) -> Video | None:
        video_id = get_video_id_from_video_link(video['Link'])
        try:
            res_video = Video(
                video_id=video_id,
                video_link=video['Link'],
                viewed_at=datetime.strptime(video['Date'], "%Y-%m-%d %H:%M:%S"),
                is_transcribed_locally=False,
                is_liked=video['Link'] in liked_links
            )

            video_metadata = self.get_video_metadata_from_link(video['Link'])

            if video_metadata is None:
                res_video.reason_for_skip_processing = ReasonsForSkipProcessing.NO_METADATA.value
                return res_video

            res_video.metadata = video_metadata
            caption_infos = self.get_caption_infos(video_metadata)
            caption_text = self.extract_caption_from_infos(caption_infos)

            if caption_text is not None:
                res_video.subtitle_text = caption_text
            else:
                # TODO: Обработка через асинхронную очередь выделения текста посредством Whisper.
                #  Здесь ожидается запись в очередь
                res_video.subtitle_text = None
                res_video.reason_for_skip_processing = ReasonsForSkipProcessing.NO_CAPTION_TEXT.value

            return res_video
        except Exception as e:
            logger.error(f"Video skipped. Failed to process {video.get('Link', 'UNKNOWN')}: {e}", exc_info=True)
            return None

    def collect_video_from_user_metadata_parallel(self, batch_size, workers):
        batch = []
        logger.info(f"Processing for user email={self.user_data.user_email}, name={self.user_data.user_nickname}")
        logger.info(f"User history contains {len(self.user_data.history_video_list)} videos")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.process_video, video, self.user_data.liked_links): video for video in
                self.user_data.history_video_list
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                video = futures[future]
                logger.debug(f"[{idx}/{len(self.user_data.history_video_list)}] Completed {video['Link']}")
                try:
                    result = future.result()
                    if result:
                        batch.append(result)
                except Exception as e:
                    logger.error(f"Error processing future result: {e}")

                if len(batch) >= batch_size:
                    insert_videos_to_database(batch)
                    batch.clear()
        if batch:
            insert_videos_to_database(batch)
