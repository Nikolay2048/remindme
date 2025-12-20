from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Video:
    video_id: str
    video_link: str
    viewed_at: Optional[datetime] = None
    is_liked: Optional[bool] = None
    is_transcribed_locally: Optional[bool] = None
    language_code: Optional[str] = None
    metadata: Optional[dict] = None
    subtitle_text: Optional[str] = None
    reason_for_skip_processing: Optional[str] = None
