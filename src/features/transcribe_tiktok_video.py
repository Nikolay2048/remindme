import logging
import os
import subprocess
from pathlib import Path

import pyktok as pyk
import whisper

from src.db.repository import VideoRepository
from src.models.enums import ReasonsForSkipProcessing
from src.models.video import Video
from src.utils.commons import wait_for_file

logger = logging.getLogger(__name__)

BASE_DIR = Path.cwd()
VIDEO_DIR = BASE_DIR / "data" / "external" / "videos"
AUDIO_DIR = BASE_DIR / "data" / "external" / "audios"

MODEL = whisper.load_model(os.getenv("WHISPER_MODEL"))
IS_DELETE_VIDEOS = os.getenv("IS_DELETE_VIDEOS_AFTER_SPEACH_RECOGNITIONS", "false").lower() == "true"
BATCH_SIZE = int(os.getenv("BATCH_SIZE_FOR_PROCESSING"))

for path in [VIDEO_DIR, AUDIO_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def process_video_without_captions():
    """
    Обрабатывает видео без субтитров из БД, добавляет субтитры.
    Returns: Обновленное состояние в БД с добовленными субтитрами

    """
    offset = 0
    total_processed = 0
    while True:
        videos = []
        batch = VideoRepository.get_skipped_videos_by_reason(ReasonsForSkipProcessing.NO_CAPTION_TEXT.value, BATCH_SIZE,
                                                             offset)
        if not batch:
            break
        for skipped_video in batch:
            videos.append(Video(
                video_id=skipped_video['video_id'],
                video_link=skipped_video['video_link'],
                is_transcribed_locally=True
            ))
        recognize_speach_tiktok_videos(videos)
        VideoRepository.update_videos_in_database(videos)
        total_processed += len(batch)
        offset += BATCH_SIZE
        logger.info(f"Processing: {total_processed} records from skipped links")


def download_video(video_link) -> str:
    """Сохранение видео по ссылке. Возвращает имя файла с видео"""
    file_name = video_link.replace("https://www.tiktokv.com/", "").replace("/", "_") + ".mp4"
    temp_file = Path.cwd() / file_name
    target_file = VIDEO_DIR / file_name

    try:
        pyk.save_tiktok(video_link, True)
        wait_for_file(temp_file)
        if not temp_file.exists():
            raise FileNotFoundError(f"Expected downloaded file {temp_file} not found")
        temp_file.rename(target_file)
    except Exception as e:
        logger.error(f"Error while downloading video {video_link}: {e}", exc_info=True)
        raise

    return target_file.name


def extract_audio(video_name_file, audio_file):
    try:
        subprocess.run([
            'ffmpeg', '-i', str(VIDEO_DIR / video_name_file), '-vn',
            str(AUDIO_DIR / audio_file)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed for {video_name_file}: {e.stderr.decode()}")
        raise


def transcribe_audio(audio_file) -> tuple[str | None, str | None]:
    """
    :param audio_file:
    :return: tuple: A tuple containing the transcript (str or None) and an error message (str or None).
    """
    try:
        transcript = MODEL.transcribe(str(AUDIO_DIR / audio_file), language='en').get("text", None)
        return transcript, None
    except Exception as e:
        logger.error(f"Error while transcribing audio {audio_file}: {e}")
        return None, str(e)


def process_video(video):
    video_file_name = None
    audio_file = None
    try:
        video_file_name = download_video(video.video_link)
        audio_file = Path(video_file_name).with_suffix(".mp3")
        extract_audio(video_file_name, audio_file)
        transcript, error_transcribing = transcribe_audio(audio_file)
        if transcript:
            video.subtitle_text = transcript
        else:
            video.subtitle_text = None
            video.reason_for_skip_processing = error_transcribing
    except Exception as e:
        logger.error(e)
        video.reason_for_skip_processing = str(e)
    finally:
        if IS_DELETE_VIDEOS:
            if video_file_name:
                video_path = VIDEO_DIR / video_file_name
                video_path.unlink(missing_ok=True)
            if audio_file:
                audio_path = AUDIO_DIR / audio_file
                audio_path.unlink(missing_ok=True)


def recognize_speach_tiktok_videos(videos):
    for video in videos:
        process_video(video)

    # with ProcessPoolExecutor(max_workers=4, initializer=init_worker) as executor:
    #     future_to_video = {executor.submit(process_video, vid): vid for vid in videos}
    #     for future in tqdm(as_completed(future_to_video), total=len(videos), desc="Processing videos"):
    #         vid = future_to_video[future]
    #         try:
    #             future.result()
    #         except Exception as exc:
    #             logger.error(f"Video {vid.video_link} raised: {exc}")
