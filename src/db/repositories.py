import logging
from typing import Any, Dict, List, Sequence

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor, execute_batch, Json

from src.db.postgres import get_conn
from src.models.user_data import UserData
from src.models.video import Video, UserVideo
from src.models.word import UserWord
from src.services.word_processor import WordProcessor
from src.utils.text_norm import normalize_vocab_key

logger = logging.getLogger(__name__)
load_dotenv()


class UserRepository:
    @staticmethod
    def add_user(user: UserData) -> None:
        sql = """
              INSERT INTO users (id, username, tg_username, email)
              VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING; \
              """
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user.id, user.user_name, user.tg_username, user.email))


class VideoRepository:
    @staticmethod
    def get_existing_video_ids() -> List[int]:
        sql = "SELECT video_id FROM tik_tok_video;"
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return [r[0] for r in cur.fetchall()]

    @staticmethod
    def get_user_video_ids(user_id: int) -> List[int]:
        sql = "SELECT video_id FROM user_video WHERE user_id = %s;"
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return [r[0] for r in cur.fetchall()]

    @staticmethod
    def insert_user_videos(user_id: int, videos: Sequence[UserVideo]) -> None:
        if not videos:
            logger.debug("No videos to insert. Skipping.")
            return

        query = f"""
                    INSERT INTO user_video
                    (user_id, video_id, viewed_at, is_liked)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, video_id, viewed_at) DO NOTHING;
                """

        rows = [
            (
                user_id,
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
            logger.exception("Error inserting user videos into postgres: %s", e)
            raise

    @staticmethod
    def insert_tik_tok_videos(videos: Sequence[Video]) -> None:
        """
        rows: [{"video_id":..., "video_link":..., "subtitle_text":..., ...}]
        """
        if not videos:
            return

        sql = """
              INSERT INTO tik_tok_video
              (video_id, video_link, subtitle_text, is_transcribed_locally, language_code, metadata,
               reason_for_skip_processing)
              VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (video_id) DO NOTHING; \
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

        with get_conn() as conn:
            with conn.cursor() as cur:
                execute_batch(cur, sql, rows, page_size=200)

    @staticmethod
    def get_skipped_videos_by_reason(reason: str, batch_size: int = 500, offset: int = 0) -> List[Dict[str, Any]]:
        sql = """
              SELECT video_id, video_link
              FROM tik_tok_video
              WHERE reason_for_skip_processing = %s
              ORDER BY video_id ASC
                  LIMIT %s
              OFFSET %s; \
              """
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (reason, batch_size, offset))
                return list(cur.fetchall())

    @staticmethod
    def update_videos_in_database(rows: Sequence[Dict[str, Any]]) -> None:
        """
        rows: [{"video_id":..., "subtitle_text":..., "is_transcribed_locally":..., "reason_for_skip_processing":...}]
        """
        if not rows:
            return

        sql = """
              UPDATE tik_tok_video
              SET subtitle_text              = %s,
                  is_transcribed_locally     = %s,
                  reason_for_skip_processing = %s
              WHERE video_id = %s; \
              """

        params = []
        for r in rows:
            params.append((
                r.get("subtitle_text"),
                bool(r.get("is_transcribed_locally", False)),
                r.get("reason_for_skip_processing"),
                r["video_id"],
            ))

        with get_conn() as conn:
            with conn.cursor() as cur:
                execute_batch(cur, sql, params, page_size=200)


class UserVocabularyRepository:
    @staticmethod
    def load_user_vocab(user_id: int) -> Dict[str, Dict[str, Any]]:
        sql = """
              SELECT text,
                     item_type,
                     difficulty_level_cefr,
                     source,
                     contexts,
                     passive_knowledge,
                     active_knowledge,
                     mastery,
                     status,
                     seen_count,
                     correct_count,
                     wrong_count
              FROM user_vocabulary
              WHERE user_id = %s; \
              """
        out: Dict[str, Dict[str, Any]] = {}
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                for row in cur.fetchall():
                    text = row[0]
                    out[text] = {
                        "text": row[0],
                        "item_type": row[1],
                        "difficulty_level_cefr": row[2],
                        "source": row[3] or [],
                        "contexts": row[4] or [],
                        "passive_knowledge": row[5],
                        "active_knowledge": row[6],
                        "mastery": row[7],
                        "status": row[8],
                        "seen_count": row[9],
                        "correct_count": row[10],
                        "wrong_count": row[11],
                    }
        return out

    @staticmethod
    def upsert_many(user_id: int, upserts: list[dict], page_size: int = 500) -> None:
        """
        upserts item example:
          {
            "text": "recommend",
            "item_type": "WORD",
            "difficulty_level_cefr": "B1",
            "source": ["lesson:123"],
            "contexts": ["..."],
            "passive_knowledge": True,
            "active_knowledge": False,
            "mastery_after": 10,
            "status_after": "learning",
            "seen_inc": 1,
            "correct_inc": 0,
            "wrong_inc": 0,
            "mastery_delta": 2
          }
        """
        if not upserts:
            return

        sql = """
              INSERT INTO user_vocabulary (user_id, text, lemma, item_type, difficulty_level_cefr,
                                           source, contexts,
                                           passive_knowledge, active_knowledge,
                                           mastery, status,
                                           seen_count, correct_count, wrong_count,
                                           last_seen_at,
                                           created_at, updated_at)
              VALUES (%(user_id)s, %(text)s, %(lemma)s, %(item_type)s, %(difficulty_level_cefr)s,
                      %(source)s, %(contexts)s,
                      %(passive_knowledge)s, %(active_knowledge)s,
                      %(mastery_after)s, %(status_after)s,
                      %(seen_inc)s, %(correct_inc)s, %(wrong_inc)s,
                      now(),
                      now(), now()) ON CONFLICT (user_id, text)
        DO
              UPDATE SET
                  item_type = EXCLUDED.item_type,
                  difficulty_level_cefr = COALESCE (user_vocabulary.difficulty_level_cefr, EXCLUDED.difficulty_level_cefr),
                  source = (
                  SELECT ARRAY(
                  SELECT DISTINCT s
                  FROM unnest(user_vocabulary.source || EXCLUDED.source) s
                  )
                  ),

                  contexts = (
                  SELECT ARRAY(
                  SELECT DISTINCT c
                  FROM unnest(user_vocabulary.contexts || EXCLUDED.contexts) c
                  )
                  ),

                  passive_knowledge = EXCLUDED.passive_knowledge OR user_vocabulary.passive_knowledge,
                  active_knowledge = EXCLUDED.active_knowledge OR user_vocabulary.active_knowledge,

                  mastery = GREATEST(0, LEAST(100, user_vocabulary.mastery + (%(mastery_delta)s))),
                  status = EXCLUDED.status,

                  seen_count = user_vocabulary.seen_count + EXCLUDED.seen_count,
                  correct_count = user_vocabulary.correct_count + EXCLUDED.correct_count,
                  wrong_count = user_vocabulary.wrong_count + EXCLUDED.wrong_count,

                  last_seen_at = now(),
                  updated_at = now(); \
              """

        params = []
        wp = WordProcessor()
        for u in upserts:
            params.append({
                "user_id": user_id,
                "text": u["text"],
                "lemma": " ".join(wp.get_lexemes(u["text"])),
                "item_type": u.get("item_type", "WORD"),
                "difficulty_level_cefr": u.get("difficulty_level_cefr", "UNKNOWN"),
                "source": u.get("source") or [],
                "contexts": u.get("contexts") or [],
                "passive_knowledge": bool(u.get("passive_knowledge", True)),
                "active_knowledge": bool(u.get("active_knowledge", False)),
                "mastery_after": int(u.get("mastery_after", 0)),
                "status_after": u.get("status_after", "new"),
                "seen_inc": int(u.get("seen_inc", 0)),
                "correct_inc": int(u.get("correct_inc", 0)),
                "wrong_inc": int(u.get("wrong_inc", 0)),
                "mastery_delta": int(u.get("mastery_delta", 0)),
            })

        with get_conn() as conn:
            with conn.cursor() as cur:
                execute_batch(cur, sql, params, page_size=page_size)

    @staticmethod
    def insert_or_update_word(user_id: int, words: list[UserWord], page_size: int = 500) -> None:
        """
        Upsert words coming from YandexTranslatorProcessor (UserWord dataclass).

        Stores into user_vocabulary:
          - text (key, lower)
          - lemma
          - translation_text / translation_lemma
          - collection_name
          - source[] merged
          - seen_count += 1, last_seen_at=now()
        """
        if not words:
            logger.debug("No words to insert. Skipping.")
            return

        sql = """
              INSERT INTO user_vocabulary (user_id, \
                                           text, \
                                           item_type, \
                                           difficulty_level_cefr, \
                                           source, \
                                           contexts, \
                                           passive_knowledge, \
                                           active_knowledge, \
                                           mastery, \
                                           status, \
                                           seen_count, \
                                           correct_count, \
                                           wrong_count, \
                                           last_seen_at, \
                                           updated_at, \
                                           collection_name, \
                                           lemma, \
                                           translation_text, \
                                           translation_lemma, \
                                           created_at)
              VALUES (%(user_id)s, \
                      %(text)s, \
                      %(item_type)s, \
                      %(difficulty_level_cefr)s, \
                      ARRAY[%(source)s], \
                      %(contexts)s, \
                      %(passive_knowledge)s, \
                      %(active_knowledge)s, \
                      %(mastery)s, \
                      %(status)s, \
                      %(seen_count)s, \
                      %(correct_count)s, \
                      %(wrong_count)s, \
                      now(), \
                      now(), \
                      %(collection_name)s, \
                      %(lemma)s, \
                      %(translation_text)s, \
                      %(translation_lemma)s, \
                      %(created_at)s) ON CONFLICT (user_id, text)
        DO \
              UPDATE SET
                  collection_name = COALESCE (user_vocabulary.collection_name, EXCLUDED.collection_name), \
                  lemma = COALESCE (user_vocabulary.lemma, EXCLUDED.lemma), \
                  translation_text = COALESCE (user_vocabulary.translation_text, EXCLUDED.translation_text), \
                  translation_lemma = COALESCE (user_vocabulary.translation_lemma, EXCLUDED.translation_lemma), \
                  source = ( \
                  SELECT ARRAY( \
                  SELECT DISTINCT s \
                  FROM unnest(user_vocabulary.source || EXCLUDED.source) s \
                  ) \
                  ), \
                  contexts = ( \
                  SELECT ARRAY( \
                  SELECT DISTINCT c \
                  FROM unnest( \
                  COALESCE (user_vocabulary.contexts, '{}'::text[]) \
                  || COALESCE (EXCLUDED.contexts, '{}'::text[]) \
                  ) c \
                  ) \
                  ), \
                  passive_knowledge = user_vocabulary.passive_knowledge OR EXCLUDED.passive_knowledge, \
                  active_knowledge = user_vocabulary.active_knowledge OR EXCLUDED.active_knowledge, \
                  seen_count = user_vocabulary.seen_count + EXCLUDED.seen_count, \
                  correct_count = user_vocabulary.correct_count + EXCLUDED.correct_count, \
                  wrong_count = user_vocabulary.wrong_count + EXCLUDED.wrong_count, \
                  last_seen_at = now(), \
                  updated_at = now(); \
              """

        params = []
        for w in words:
            raw_text = (w.text or "").strip()
            if not raw_text:
                continue

            text_key = normalize_vocab_key(raw_text)
            item_type = "PHRASE" if " " in text_key else "WORD"

            params.append({
                "user_id": user_id,
                "text": text_key,
                "item_type": item_type,
                "difficulty_level_cefr": "UNKNOWN",

                "source": w.source.value,
                "contexts": [],

                "passive_knowledge": True,
                "active_knowledge": False,
                "mastery": 0,
                "status": "new",

                "seen_count": 1,
                "correct_count": 0,
                "wrong_count": 0,

                "collection_name": w.collection_name,
                "lemma": w.lemma,
                "translation_text": w.translation_text,
                "translation_lemma": w.translation_lemma,
                "created_at": w.creation_datetime,
            })

        with get_conn() as conn:
            with conn.cursor() as cur:
                execute_batch(cur, sql, params, page_size=page_size)


class LessonRepository:
    @staticmethod
    def create_lesson(
            user_id: int,
            *,
            title: str | None = None,
            source: str | None = None,
            source_uri: str | None = None,
            language_mode: str = "ru+en",
            duration_sec: int | None = None,
            started_at=None,
            metadata: dict | None = None,
    ) -> int:
        sql = """
              INSERT INTO lesson (user_id, title, source, source_uri, language_mode, duration_sec, started_at, metadata)
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb) RETURNING id; \
              """
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    user_id, title, source, source_uri, language_mode, duration_sec, started_at,
                    Json(metadata) if metadata else None
                ))
                return cur.fetchone()[0]

    @staticmethod
    def update_lesson_asr_payload(
            lesson_id: int,
            *,
            teacher_full_text: str | None,
            student_full_text: str | None,
            dialog_utterances_jsonb: list[dict] | None,
            talk_metrics: dict | None,
            report_json: dict | None = None,
            materials_text: str | None = None,
    ) -> None:
        tm = talk_metrics or {}

        sql = """
              UPDATE lesson
              SET teacher_full_text       = %s,
                  student_full_text       = %s,
                  dialog_utterances_jsonb = %s::jsonb,

            teacher_time_sec = %s, student_time_sec = %s, student_talk_ratio = %s, teacher_words_count = %s, student_words_count = %s, report_json = %s::jsonb, materials_text = %s
              WHERE id = %s; \
              """

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    teacher_full_text,
                    student_full_text,
                    Json(dialog_utterances_jsonb) if dialog_utterances_jsonb is not None else None,

                    tm.get("teacher_speaking_time_sec"),
                    tm.get("student_speaking_time_sec"),
                    tm.get("student_speaking_ratio"),
                    tm.get("teacher_words_count"),
                    tm.get("student_words_count"),

                    Json(report_json) if report_json else None,
                    materials_text,
                    lesson_id,
                ))

    @staticmethod
    def insert_turns(lesson_id: int, turns: list[dict], page_size: int = 500) -> None:
        if not turns:
            return

        sql = """
              INSERT INTO lesson_turn
              (lesson_id, turn_id, role, start_s, end_s, text, is_question, language_hint, stats)
              VALUES (%(lesson_id)s, %(turn_id)s, %(role)s, %(start_s)s, %(end_s)s, %(text)s, %(is_question)s,
                      %(language_hint)s, %(stats)s::jsonb) ON CONFLICT (lesson_id, turn_id) DO
              UPDATE SET
                  role = EXCLUDED.role,
                  start_s = EXCLUDED.start_s,
                  end_s = EXCLUDED.end_s,
                  text = EXCLUDED.text,
                  is_question = EXCLUDED.is_question,
                  language_hint = EXCLUDED.language_hint,
                  stats = EXCLUDED.stats; \
              """

        params = []
        for t in turns:
            text = (t.get("text") or "").strip()
            if not text:
                continue
            params.append({
                "lesson_id": lesson_id,
                "turn_id": int(t["turn_id"]),
                "role": t.get("role"),
                "start_s": float(t["t_start_sec"]),
                "end_s": float(t["t_end_sec"]),
                "text": text,
                "is_question": bool(t.get("is_question", False)),
                "language_hint": t.get("language_hint"),
                "stats": Json(t.get("stats") or {}),
            })

        if not params:
            return

        with get_conn() as conn:
            with conn.cursor() as cur:
                execute_batch(cur, sql, params, page_size=page_size)
