import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import pyktok as pyk
import requests
from requests import RequestException

from src.db.repositories import VideoRepository
from src.models.enums import ReasonsForSkipProcessing
from src.models.user_data import UserDataTikTok
from src.models.video import Video, UserVideo
from src.utils import extract_text_from_vtt

logger = logging.getLogger(__name__)


class TikTokProcessor:

    def __init__(self, user_data_tik_tok: UserDataTikTok):
        self.user_data_tik_tok = user_data_tik_tok

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

    def process_video(self, video_link: str, existing_video) -> Video | None:
        video_id = self._get_video_id_from_video_link(video_link)
        if video_id in existing_video:
            return
        try:
            res_video = Video(
                video_id=video_id,
                video_link=video_link,
                is_transcribed_locally=False,
            )

            video_metadata = self.get_video_metadata_from_link(video_link)

            if video_metadata is None:
                res_video.reason_for_skip_processing = ReasonsForSkipProcessing.NO_METADATA.value
                return res_video

            res_video.metadata = video_metadata
            caption_infos = self.get_caption_infos(video_metadata)
            caption_text = self.extract_caption_from_infos(caption_infos)

            if caption_text is not None:
                res_video.subtitle_text = caption_text
            else:
                # TODO: Обработка через асинхронную очередь выделения текста (Rabbit\Kafka) посредством Whisper.
                #  Здесь ожидается запись в очередь
                res_video.subtitle_text = None
                res_video.reason_for_skip_processing = ReasonsForSkipProcessing.NO_CAPTION_TEXT.value

            return res_video
        except Exception as e:
            logger.error(f"Video skipped. Failed to process video by link {video_link}: {e}", exc_info=True)
            return None

    def collect_video_captions_from_user_videos(self, batch_size, workers,
                                                history_video_list: list[dict[str, Any]]) -> None:
        batch = []
        existing_video = VideoRepository.get_existing_video_ids()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.process_video, video["Link"], existing_video): video for video in
                history_video_list
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                video = futures[future]
                logger.debug(f"[{idx}/{len(history_video_list)}] Completed {video}")
                try:
                    result = future.result()
                    if result:
                        batch.append(result)
                except Exception as e:
                    logger.error(f"Error processing future result: {e}")

                if len(batch) >= batch_size:
                    VideoRepository.insert_tik_tok_videos(batch)
                    batch.clear()
        if batch:
            VideoRepository.insert_tik_tok_videos(batch)

    def _get_video_id_from_video_link(self, video_link: str) -> int:
        return int(video_link.split("/")[-2])

    def update_user_video_history(self, user_id: int, user_data_tik_tok: UserDataTikTok):
        logger.info(f"User history contains {len(user_data_tik_tok.history_video_list)} videos")
        batch = []
        user_videos = VideoRepository.get_user_video_ids(user_id)
        for video_data in user_data_tik_tok.history_video_list:
            video_id = self._get_video_id_from_video_link(video_data['Link'])
            if video_id in user_videos:
                continue
            batch.append(UserVideo(
                video_id=video_id,
                viewed_at=datetime.strptime(video_data['Date'], "%Y-%m-%d %H:%M:%S"),
                is_liked=video_data['Link'] in user_data_tik_tok.liked_links,
            ))
            if len(batch) >= 100:
                VideoRepository.insert_user_videos(user_id, batch)
                batch.clear()
        if batch:
            VideoRepository.insert_user_videos(user_id, batch)
