import os
import requests

ASR_URL = os.getenv("ASR_URL", "http://localhost:8000")

def transcribe_lesson(video_path: str) -> dict:
    with open(video_path, "rb") as f:
        r = requests.post(
            f"{ASR_URL}/transcribe",
            files={"file": ("lesson.mp4", f, "video/mp4")},
            timeout=60 * 60
        )
    r.raise_for_status()
    print(r.text)
    return r.json()

transcribe_lesson("video.mp4")