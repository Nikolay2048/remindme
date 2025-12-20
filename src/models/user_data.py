import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


class UserDataTikTok:

    def __init__(self, user_data_path_arch: str):
        user_data = self._extract_user_info_file(user_data_path_arch)

        self.history_video_list = user_data.get("Your Activity", {}).get("Watch History", {}).get("VideoList",
                                                                                                  [])
        self.user_email = user_data.get("Profile", {}).get("Profile Info", {}).get("ProfileMap", {}).get(
            "emailAddress", None)
        self.user_nickname = user_data.get("Profile", {}).get("Profile Info", {}).get("ProfileMap", {}).get(
            "userName", None)
        self.liked_videos = user_data.get("Your Activity", {}).get("Like List", {}).get("VideoList", [])
        self.liked_links = {item["Link"] for item in self.liked_videos}

    def _extract_user_info_file(self, user_data_path_arch: str):
        with zipfile.ZipFile(user_data_path_arch, 'r') as zip_ref:
            file_list = zip_ref.namelist()

            if len(file_list) != 1:
                raise ValueError(f"Expected 1 file in archive, but found {len(file_list)}")

            extracted_file = file_list[0]
            zip_ref.extract(extracted_file)

            with open(extracted_file, 'r', encoding='utf-8') as data_file:
                user_data = json.load(data_file)
                return user_data


@dataclass
class UserData:
    id: int
    user_name: str
    tg_username: str
    email: str
    created_at: datetime = datetime.now()
    last_visit_at: datetime = datetime.now()
