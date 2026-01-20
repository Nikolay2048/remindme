from __future__ import annotations

import datetime
import json
import os
from typing import Dict, Any, Optional

import torch

from src.db.repositories import LessonRepository, UserVocabularyRepository
from src.services.word_processor import WordProcessor
from .audio_extract import AudioExtractor
from .report import build_lesson_report
from .transcribe import LessonTranscriber
from .vocab import VocabAnalyzer, CefrLexicon


class LessonProcessor:
    def __init__(
            self,
            *,
            whisper_model_size: str = "small",
            device: str =  os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
            compute_type: str = os.getenv("COMPUTE_TYPE", "float16" if torch.cuda.is_available() else "int8"),
            language: str = "en",
            utterance_gap_ms: int = 1000,
            low_prob: float = 0.55,
            high_prob: float = 0.75,
            long_pause_ms: int = 500,
            cefr_dict: Optional[Dict[str, str]] = None,
    ):
        self.audio = AudioExtractor()
        self.transcriber = LessonTranscriber(
            whisper_model_size=whisper_model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            utterance_gap_ms=utterance_gap_ms,
        )
        self.wp = WordProcessor(keep_stopwords=False, min_len=2)
        self.cefr = CefrLexicon(cefr_dict) if cefr_dict else None

        self.vocab = VocabAnalyzer(
            wp=self.wp,
            cefr=self.cefr,
            low_prob=low_prob,
            high_prob=high_prob,
            long_pause_ms=long_pause_ms,
            phrase_whitelist=None,
            phrase_min_count=10 ** 9,  # phrases off in MVP
        )

    def process_video_to_db(
            self,
            *,
            user_id: int,
            video_path: str,
            out_dir: str = "_lesson",
            title: Optional[str] = None,
            source: str = "file",
            materials_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        # 0) load existing vocab from DB
        existing_user_vocab = UserVocabularyRepository.load_user_vocab(user_id)

        # 1) create lesson row
        lesson_db_id = LessonRepository.create_lesson(
            user_id=user_id,
            title=title,
            source=source,
            source_uri=video_path,
            language_mode="ru+en",
            duration_sec=None,
            started_at=None,
            metadata={"pipeline": "lesson_analyzer_v1"},
        )

        # local ids for filesystem/report
        lesson_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        lesson_dir = os.path.join(out_dir, f"lesson_{lesson_datetime}")
        os.makedirs(lesson_dir, exist_ok=True)

        # 2) extract tracks
        tracks = self.audio.extract_three_tracks(video_path=video_path, out_dir=lesson_dir)

        # 3) transcribe
        tr = self.transcriber.transcribe_two_roles(
            teacher_audio_path=tracks.teacher_wav,
            student_audio_path=tracks.student_wav,
        )

        # 4) talk metrics
        talk_metrics = self._compute_talk_metrics(
            teacher_utts=tr["teacher_utterances"],
            student_utts=tr["student_utterances"],
            teacher_words=tr["teacher_words"],
            student_words=tr["student_words"],
        )

        # 5) mind map
        mind_map = self._build_mind_map(
            teacher_utts=tr["teacher_utterances"],
            student_utts=tr["student_utterances"],
        )

        # 6) vocab analysis + upserts
        vocab_res = self.vocab.analyze(
            lesson_id=str(lesson_db_id),  # lesson id can be DB id here for source tags
            student_words=tr["student_words"],
            teacher_words=tr["teacher_words"],
            student_utts=tr["student_utterances"],
            existing_user_vocab=existing_user_vocab,
        )

        # 7) build report
        report = build_lesson_report(
            schema_version="1.0",
            lesson_id=str(lesson_db_id),
            user_id=user_id,
            source={
                "video_path": video_path,
                "language": "en",
                "materials_provided": bool(materials_text),
            },
            talk_metrics=talk_metrics,
            vocabulary_block=vocab_res["vocabulary"],
            mind_map=mind_map,
            action_items=vocab_res["action_items"],
        )

        report_path = os.path.join(lesson_dir, "lesson_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 8) write to DB: turns + lesson payload + vocab
        LessonRepository.insert_turns(lesson_db_id, tr["dialog_utterances_jsonb"])
        LessonRepository.update_lesson_asr_payload(
            lesson_db_id,
            teacher_full_text=tr["teacher_full_text"],
            student_full_text=tr["student_full_text"],
            dialog_utterances_jsonb=tr["dialog_utterances_jsonb"],
            talk_metrics=talk_metrics,
            report_json=report,
            materials_text=materials_text,
        )
        UserVocabularyRepository.upsert_many(user_id, vocab_res["upserts"])

        return {
            "lesson_db_id": lesson_db_id,
            "paths": {
                "lesson_dir": lesson_dir,
                "report_json": report_path,
                "student_wav": tracks.student_wav,
                "teacher_wav": tracks.teacher_wav,
                "mixed_wav": tracks.mixed_wav,
            },
            "aggregates": vocab_res["aggregates"],
        }

    def _compute_talk_metrics(self, *, teacher_utts, student_utts, teacher_words, student_words) -> Dict[str, Any]:
        def utt_time(utts) -> float:
            return sum(max(0.0, u.end - u.start) for u in utts)

        teacher_time = utt_time(teacher_utts)
        student_time = utt_time(student_utts)
        total = max(1e-6, teacher_time + student_time)

        return {
            "teacher_speaking_time_sec": int(round(teacher_time)),
            "student_speaking_time_sec": int(round(student_time)),
            "student_speaking_ratio": round(student_time / total, 3),
            "teacher_words_count": len(teacher_words),
            "student_words_count": len(student_words),
        }

    def _build_mind_map(self, *, teacher_utts, student_utts) -> Dict[str, Any]:
        timeline = sorted(list(teacher_utts) + list(student_utts), key=lambda u: (u.start, u.end))
        if not timeline:
            return {"segments": []}

        segments = []
        block = []
        block_id = 1

        def flush():
            nonlocal block_id
            if not block:
                return
            t0 = block[0].start
            t1 = block[-1].end
            student_time = sum((u.end - u.start) for u in block if u.role == "student")
            total_time = max(1e-6, sum((u.end - u.start) for u in block))
            snippet = " ".join([u.text for u in block if u.role == "student"][:2]).strip()
            if not snippet:
                snippet = (block[0].text or "")[:200]

            segments.append({
                "segment_id": block_id,
                "t_start_sec": round(float(t0), 2),
                "t_end_sec": round(float(t1), 2),
                "keywords": [],
                "student_speaking_ratio": round(float(student_time / total_time), 3),
                "snippet": snippet,
            })
            block_id += 1
            block.clear()

        for u in timeline:
            block.append(u)
            if len(block) >= 8:
                flush()
        flush()

        return {"segments": segments}
