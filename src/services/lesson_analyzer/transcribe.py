from __future__ import annotations

import logging
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from faster_whisper import WhisperModel

from .models import WordToken, Utterance, Role
from ..word_processor import detect_lang

logger = logging.getLogger(__name__)


class LessonTranscriber:
    def __init__(
            self,
            *,
            whisper_model_size: str,
            device: str,
            compute_type: str,
            language: str = "en",
            utterance_gap_ms: int = 1000,
    ):
        self.model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
        self.language = language
        self.utterance_gap_ms = utterance_gap_ms

    def transcribe_two_roles(
            self,
            *,
            teacher_audio_path: str,
            student_audio_path: str,
    ) -> Dict[str, Any]:
        teacher_words, teacher_full_text, teacher_utts = self._transcribe_role(teacher_audio_path, role="teacher")
        student_words, student_full_text, student_utts = self._transcribe_role(student_audio_path, role="student")

        # build combined timeline for DB/jsonb
        dialog_jsonb = self._build_dialog_jsonb(teacher_utts, student_utts)

        return {
            "teacher_words": teacher_words,
            "student_words": student_words,
            "teacher_full_text": teacher_full_text,
            "student_full_text": student_full_text,
            "teacher_utterances": teacher_utts,
            "student_utterances": student_utts,
            "dialog_utterances_jsonb": dialog_jsonb,
        }

    def _transcribe_role(self, audio_path: str, role: Role) -> Tuple[List[WordToken], str, List[Utterance]]:
        logger.info(f"Transcribing {audio_path}")
        segments, _info = self.model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        logger.info(f"Finished transcribing")
        words: List[WordToken] = []
        text_parts: List[str] = []
        prev_end: Optional[float] = None

        for seg in segments:
            if seg.text:
                text_parts.append(seg.text.strip())

            for w in (seg.words or []):
                token = (w.word or "").strip()
                if not token:
                    continue

                pause_ms = None
                if prev_end is not None and w.start is not None:
                    pause_ms = int(max(0.0, (w.start - prev_end) * 1000))
                prev_end = float(w.end) if w.end is not None else prev_end

                prob = None
                if hasattr(w, "probability") and w.probability is not None:
                    prob = float(w.probability)

                words.append(WordToken(
                    text=token.strip().lower(),
                    start=float(w.start or 0.0),
                    end=float(w.end or 0.0),
                    prob=prob,
                    pause_ms=pause_ms,
                    role=role,
                ))

        full_text = " ".join(text_parts).strip()
        utterances = self._words_to_utterances(words, gap_ms=self.utterance_gap_ms)
        return words, full_text, utterances

    def _words_to_utterances(self, words: List[WordToken], gap_ms: int) -> List[Utterance]:
        if not words:
            return []

        utts: List[Utterance] = []
        cur_tokens: List[WordToken] = []

        def flush():
            nonlocal cur_tokens
            if not cur_tokens:
                return
            start = cur_tokens[0].start
            end = cur_tokens[-1].end
            text = " ".join(t.text for t in cur_tokens).strip()
            probs = [t.prob for t in cur_tokens if t.prob is not None]
            avg_prob = round(mean(probs), 3) if probs else None
            utts.append(Utterance(
                role=cur_tokens[0].role,
                start=float(start),
                end=float(end),
                text=text,
                avg_prob=avg_prob,
                words_count=len(cur_tokens),
            ))
            cur_tokens = []

        for w in words:
            if not cur_tokens:
                cur_tokens.append(w)
                continue

            gap = 0
            prev = cur_tokens[-1]
            gap = int(max(0.0, (w.start - prev.end) * 1000))

            if gap >= gap_ms:
                flush()
                cur_tokens.append(w)
            else:
                cur_tokens.append(w)

        flush()
        return utts

    def _build_dialog_jsonb(self, teacher_utts: List[Utterance], student_utts: List[Utterance]) -> List[dict]:
        timeline = sorted(list(teacher_utts) + list(student_utts), key=lambda u: (u.start, u.end))
        out: List[dict] = []
        for i, u in enumerate(timeline, start=1):
            text = (u.text or "").strip()
            lang = detect_lang(text)  # "en"/"ru"/None
            out.append({
                "turn_id": i,
                "role": u.role,
                "t_start_sec": round(float(u.start), 2),
                "t_end_sec": round(float(u.end), 2),
                "text": text,
                "is_question": text.endswith("?"),
                "language_hint": lang,
                "stats": {
                    "avg_asr_prob": u.avg_prob,
                    "words_count": u.words_count,
                }
            })
        return out
