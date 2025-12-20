import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor, execute_batch

from src.models.user_data import UserData
from src.models.video import Video, UserVideo

logger = logging.getLogger(__name__)
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

USER_TABLE = os.getenv("DB_USERS_TABLE")
USER_VIDEO_TABLE = os.getenv("DB_USER_VIDEO_TABLE")
VIDEOS_TABLE = os.getenv("DB_TIK_TOK_VIDEOS_TABLE")


def _require_table(name: Optional[str], env_name: str) -> str:
    if not name:
        raise RuntimeError(f"Table name is not set. Provide env var {env_name}.")
    return name


@contextmanager
def get_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class UserRepository:
    @staticmethod
    def add_user(user: UserData) -> None:
        """
        Insert or update user by primary key id.
        """
        query = f"""
            INSERT INTO {USER_TABLE} (id, username, tg_username, email)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """

        params = (
            user.id,
            user.user_name,
            None,  # TODO: tg_username отсутствует в модели
            user.email,
        )

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
        except psycopg2.Error as e:
            logger.exception("Error inserting user in postgres: %s", e)
            raise


class VideoRepository:
    """
    Общий репозиторий абстракций video TikTok.
    """

    @staticmethod
    def get_existing_video_ids() -> List[str]:
        table_name = _require_table(VIDEOS_TABLE, "DB_TIK_TOK_VIDEOS_TABLE (or DB_TABLE)")
        query = f"SELECT video_id FROM {table_name};"

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()
                    return [r[0] for r in rows]
        except psycopg2.Error as e:
            logger.exception("Error selecting existing videos from postgres: %s", e)
            raise

    @staticmethod
    def get_user_video_ids(user_id: int) -> List[str]:
        query = f"SELECT video_id FROM {USER_VIDEO_TABLE} WHERE user_id = %s;"

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (user_id,))
                    rows = cur.fetchall()
                    return [r[0] for r in rows]
        except psycopg2.Error as e:
            logger.exception("Error selecting user videos from postgres: %s", e)
            raise

    @staticmethod
    def insert_user_videos(videos: Sequence[UserVideo]) -> None:
        if not videos:
            logger.debug("No videos to insert. Skipping.")
            return

        table_name = _require_table(USER_VIDEO_TABLE, "DB_TIK_TOK_VIDEOS_TABLE (or DB_TABLE)")
        query = f"""
                INSERT INTO {table_name}
                (user_id, video_id, viewed_at, is_liked)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, video_id, viewed_at) DO NOTHING;
            """

        rows = [
            (
                v.user_id,
                v.video_id,
                v.viewed_at,
                v.is_liked
            )
            for v in videos
        ]

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    execute_batch(cur, query, rows, page_size=100)
        except psycopg2.Error as e:
            logger.exception("Error inserting videos into postgres: %s", e)
            raise

    @staticmethod
    def insert_tik_tok_videos(videos: Sequence[Video]) -> None:
        if not videos:
            logger.debug("No videos to insert. Skipping.")
            return

        table_name = _require_table(VIDEOS_TABLE, "DB_TIK_TOK_VIDEOS_TABLE (or DB_TABLE)")
        query = f"""
            INSERT INTO {table_name}
            (video_id, video_link, subtitle_text, is_transcribed_locally,
             language_code, metadata, reason_for_skip_processing)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (video_id) DO NOTHING;
        """

        rows = [
            (
                v.video_id,
                v.video_link,
                v.subtitle_text,
                v.is_transcribed_locally,
                v.language_code,
                psycopg2.extras.Json(v.metadata) if getattr(v, "metadata", None) else None,
                v.reason_for_skip_processing,
            )
            for v in videos
        ]

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    execute_batch(cur, query, rows, page_size=100)
        except psycopg2.Error as e:
            logger.exception("Error inserting videos into postgres: %s", e)
            raise

    @staticmethod
    def get_skipped_videos_by_reason(
            reason: str,
            batch_size: int = 500,
            offset: int = 0,
    ) -> List[Dict[str, Any]]:
        table_name = _require_table(VIDEOS_TABLE, "DB_TIK_TOK_VIDEOS_TABLE (or DB_TABLE)")
        select_query = f"""
            SELECT video_id, video_link
            FROM {table_name}
            WHERE reason_for_skip_processing = %s
            ORDER BY video_id ASC
            LIMIT %s OFFSET %s;
        """

        try:
            with get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(select_query, (reason, batch_size, offset))
                    return list(cur.fetchall())
        except psycopg2.Error as e:
            logger.exception("Error retrieving skipped videos: %s", e)
            raise

    @staticmethod
    def update_videos_in_database(videos: Sequence[Video]) -> None:
        """
        Обновляет поля у существующих видео.
        """
        if not videos:
            logger.debug("No videos to update. Skipping.")
            return

        table_name = _require_table(VIDEOS_TABLE, "DB_TIK_TOK_VIDEOS_TABLE (or DB_TABLE)")
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
                v.video_id,
            )
            for v in videos
        ]

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    execute_batch(cur, update_query, rows, page_size=100)
                    logger.debug("Updated %s rows in %s", cur.rowcount, table_name)
        except psycopg2.Error as e:
            logger.exception("Error updating videos in postgres: %s", e)
            raise


class WordsRepository:
    pass
