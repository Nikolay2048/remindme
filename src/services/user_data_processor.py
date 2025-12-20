import logging
from datetime import datetime

from src.db.repository import VideoRepository, UserRepository
from src.models.user_data import UserData
from src.models.video import UserVideo

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
                user_id=self.user_data.id,
                video_id=video_id,
                viewed_at=datetime.strptime(video['Date'], "%Y-%m-%d %H:%M:%S"),
                is_liked=video['Link'] in self.user_data.tiktok_data.liked_links
            ))
            if len(batch) >= 100:
                VideoRepository.insert_user_videos(batch)
                batch.clear()
        if batch:
            VideoRepository.insert_user_videos(batch)

    def process_user_data(self):
        logger.info(f"Processing for user id={self.user_data.id}, email={self.user_data.email}, "
                    f"name={self.user_data.user_name}")

        UserRepository.add_user(self.user_data)

        if self.user_data.tiktok_data is not None:
            self._process_tik_tok_user_data()
