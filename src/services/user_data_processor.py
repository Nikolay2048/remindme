import logging
from datetime import datetime

from src.db.repository import VideoRepository, UserRepository, UserVocabularyRepository
from src.models.user_data import UserData
from src.models.video import UserVideo
from src.services.translator_processor import YandexTranslatorProcessor

logger = logging.getLogger(__name__)


class UserDataService:
    def __init__(self, user_data: UserData):
        self.user_data = user_data

    @staticmethod
    def get_video_id_from_video_link(video_link):
        return int(video_link.split("/")[-2])

    def get_user_video_links(self):
        return [item["Link"] for item in self.user_data.tiktok_data.history_video_list]

    def _process_tik_tok_user_data(self):
        logger.info(f"User history contains {len(self.user_data.tiktok_data.history_video_list)} videos")
        batch = []
        user_videos = VideoRepository.get_user_video_ids(self.user_data.id)
        for video in self.user_data.tiktok_data.history_video_list:
            if video in user_videos:
                continue
            video_id = self.get_video_id_from_video_link(video['Link'])
            batch.append(UserVideo(
                video_id=video_id,
                viewed_at=datetime.strptime(video['Date'], "%Y-%m-%d %H:%M:%S"),
                is_liked=video['Link'] in self.user_data.tiktok_data.liked_links
            ))
            if len(batch) >= 100:
                VideoRepository.insert_user_videos(self.user_data.id, batch)
                batch.clear()
        if batch:
            VideoRepository.insert_user_videos(self.user_data.id, batch)

    def _process_yandex_translator_user_data(self):
        translator_processor = YandexTranslatorProcessor()
        for collection_id in self.user_data.yandex_translator.collections:
            logger.info(f"Collecting words from collection={collection_id}")
            payload = translator_processor.fetch_collection(collection_id)
            logger.debug(payload)
            words = translator_processor.extract_words(payload)
            logger.info(f"Collected words pairs: {len(words)}")
            UserVocabularyRepository.insert_or_update_word(self.user_data.id, words)

    def process_user_data(self):
        logger.info(f"Processing for user id={self.user_data.id}, email={self.user_data.email}, "
                    f"name={self.user_data.user_name}")

        UserRepository.add_user(self.user_data)

        if self.user_data.tiktok_data is not None:
            self._process_tik_tok_user_data()

        if self.user_data.yandex_translator is not None:
            self._process_yandex_translator_user_data()
