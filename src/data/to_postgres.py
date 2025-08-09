import json
import logging
import os
from typing import List, Tuple

from dotenv import load_dotenv
from psycopg2.extras import Json, RealDictCursor
import psycopg2
from psycopg2.extras import execute_batch

from src.models.enums import ReasonsForSkipProcessing
from src.models.video import Video

logger = logging.getLogger(__name__)
load_dotenv()
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}


def get_skipped_videos_by_reason(reason: str, batch_size: int = 500, offset: int = 0) -> List[Tuple]:
    table_name = os.getenv("DB_TABLE")
    select_query = f"""
            SELECT video_id, video_link
        FROM {table_name}
        WHERE reason_for_skip_processing = %s
        ORDER BY video_id ASC
        LIMIT %s OFFSET %s
        """
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(select_query, (reason, batch_size, offset,))
                results = cur.fetchall()
                return results

    except psycopg2.Error as e:
        logger.error(f"Error retrieving skipped videos: {e}")
        return []


def update_videos_in_database(videos: List[Video]) -> None:
    """
    Вставляет список транскриптов в таблицу PostgreSQL.
    """
    if not videos:
        logger.debug("No videos to update. Skipping.")
        return
    table_name = os.getenv("DB_TABLE")
    update_query = f"""
        UPDATE {table_name}
        SET 
            subtitle_text = %s,
            is_transcribed_locally = %s,
            reason_for_skip_processing = %s
        WHERE video_id = %s;
        """
    rows = [
        (
            v.subtitle_text,
            v.is_transcribed_locally,
            v.reason_for_skip_processing,
            v.video_id
        )
        for v in videos
    ]

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                execute_batch(cur, update_query, rows, page_size=100)
                logger.debug(f"Updated {cur.rowcount} rows in {table_name}")
    except psycopg2.Error as e:
        logger.error(f"Error updating data in postgres: {e}")


def insert_videos_to_database(videos: List[Video]) -> None:
    """
    Обновляет список транскриптов в таблице PostgreSQL.
    """
    if not videos:
        logger.debug("No videos to insert. Skipping.")
        return
    table_name = os.getenv("DB_TABLE")
    insert_query = f"""
        INSERT INTO {table_name} 
        (video_id, video_link, viewed_at, subtitle_text, is_liked, is_transcribed_locally, 
        language_code, metadata, reason_for_skip_processing)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (video_id) DO NOTHING;
        """
    rows = [
        (
            v.video_id,
            v.video_link,
            v.viewed_at,
            v.subtitle_text,
            v.is_liked,
            v.is_transcribed_locally,
            v.language_code,
            Json(v.metadata) if v.metadata else None,
            v.reason_for_skip_processing
        )
        for v in videos
    ]
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                execute_batch(cur, insert_query, rows, page_size=100)
    except psycopg2.Error as e:
        logger.error(f"Error inserting data into postgres: {e}")
