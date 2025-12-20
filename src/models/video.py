from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Video:
    video_id: int
    video_link: str
    is_transcribed_locally: Optional[bool] = None
    language_code: Optional[str] = None
    metadata: Optional[dict] = None
    subtitle_text: Optional[str] = None
    reason_for_skip_processing: Optional[str] = None


@dataclass
class UserVideo:
    user_id: int
    video_id: int
    viewed_at: datetime
    is_liked: Optional[bool] = None
