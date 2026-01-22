from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from faster_whisper import WhisperModel

from .models import WordToken, Utterance, Role
from ..word_processor import detect_lang

logger = logging.getLogger(__name__)


@dataclass
class _SegPack:
    start: float
    end: float
    text: str
    words: list
    score: float


@dataclass
class _Chunk:
    idx: int
    src_start: float
    src_end: float
    path: str

    @property
    def dur(self) -> float:
        return max(0.0, self.src_end - self.src_start)


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
        if whisper_model_size.endswith(".en") or whisper_model_size in {"tiny.en", "base.en", "small.en", "medium.en"}:
            raise ValueError(
                f"English-only Whisper model '{whisper_model_size}' will cause RU->EN 'translation-like' output. "
                f"Use multilingual model without '.en' (e.g., 'small', 'medium', 'large-v3')."
            )

        self.model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
        self.language = language
        self.utterance_gap_ms = utterance_gap_ms

        # дробление (НЕ трогаем)
        self.split_pause_sec = 2.0
        self.silence_db = -35
        self.cut_pad_sec = 0.05
        self.min_chunk_sec = 0.25
        self.chunks_dir_name = "_asr_chunks"

        # порог фильтра “мусорных” сегментов внутри чанка
        self.min_segment_score = 0.20

    def transcribe_two_roles(
        self,
        *,
        teacher_audio_path: str,
        student_audio_path: str,
    ) -> Dict[str, Any]:
        teacher_words, teacher_full_text, teacher_utts = self._transcribe_role_stable(teacher_audio_path, role="teacher")
        student_words, student_full_text, student_utts = self._transcribe_role_stable(student_audio_path, role="student")

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

    # ============================================================
    # STABLE: split -> transcribe each chunk once -> print texts
    # ============================================================

    def _transcribe_role_stable(self, audio_path: str, role: Role) -> Tuple[List[WordToken], str, List[Utterance]]:
        logger.info(f"Splitting by pauses >= {self.split_pause_sec}s: {audio_path}")

        chunks = self._split_audio_into_speech_chunks(audio_path)
        self._print_chunks_info(chunks)

        all_words: List[WordToken] = []
        all_text_parts: List[str] = []
        prev_end_abs: Optional[float] = None

        print(f"\n========== CHUNK TEXTS ({role}) ==========")

        for ch in chunks:
            segs = self._transcribe_segments_single_pass(ch.path)

            # текст чанка
            chunk_text = " ".join(s.text.strip() for s in segs if s.text).strip()

            # печать по чанкам
            print(f"\n--- chunk #{ch.idx:04d}  [{ch.src_start:.2f}s..{ch.src_end:.2f}s]  dur={ch.dur:.2f}s ---")
            print(chunk_text)

            if chunk_text:
                all_text_parts.append(chunk_text)

            # слова чанка -> общий список (абсолютные таймкоды!)
            for seg in segs:
                for w in (seg.words or []):
                    token = (getattr(w, "word", None) or "").strip()
                    if not token:
                        continue

                    w_start = float(getattr(w, "start", 0.0) or 0.0)
                    w_end = float(getattr(w, "end", 0.0) or 0.0)

                    abs_start = ch.src_start + w_start
                    abs_end = ch.src_start + w_end

                    pause_ms = None
                    if prev_end_abs is not None:
                        pause_ms = int(max(0.0, (abs_start - prev_end_abs) * 1000))
                    prev_end_abs = abs_end

                    prob = None
                    if hasattr(w, "probability") and w.probability is not None:
                        prob = float(w.probability)

                    all_words.append(
                        WordToken(
                            text=token.strip().lower(),
                            start=abs_start,
                            end=abs_end,
                            prob=prob,
                            pause_ms=pause_ms,
                            role=role,
                        )
                    )

        all_words.sort(key=lambda t: (t.start, t.end))
        full_text = "\n".join(p for p in all_text_parts if p).strip()

        print(f"\n========== FULL TEXT ({role}) ==========\n")
        print(full_text)
        print(f"\n========== /FULL TEXT ({role}) ==========\n")

        utterances = self._words_to_utterances(all_words, gap_ms=self.utterance_gap_ms)
        return all_words, full_text, utterances

    # ============================================================
    # Single-pass ASR (no ru/en)
    # ============================================================

    def _transcribe_segments_single_pass(self, audio_path: str) -> List[_SegPack]:
        """
        Один прогон. language=None (авто), task=transcribe.
        Важные параметры для снижения "переводоподобности" и переноса контекста:
        condition_on_previous_text=False + temperature=0.
        """
        segments, _info = self.model.transcribe(
            audio_path,
            language=None,                 # авто ru/en
            task="transcribe",             # НЕ переводить
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            temperature=0.0,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=True
        )

        if _info.language not in ("ru", "en"):
            print("===== INNFFOO =====")
            print(_info)
        out: List[_SegPack] = []
        for seg in segments:
            start = float(getattr(seg, "start", 0.0) or 0.0)
            end = float(getattr(seg, "end", start) or start)
            text = (getattr(seg, "text", "") or "").strip()
            words = list(getattr(seg, "words", None) or [])
            score = self._segment_score(seg, words)

            # лёгкий фильтр мусора
            if text and score >= self.min_segment_score:
                out.append(_SegPack(start=start, end=end, text=text, words=words, score=score))

        return out

    def _segment_score(self, seg: Any, words: List[Any]) -> float:
        probs = []
        for w in words:
            p = getattr(w, "probability", None)
            if p is not None:
                probs.append(float(p))
        if probs:
            return float(mean(probs))

        avg_logprob = getattr(seg, "avg_logprob", None)
        if avg_logprob is not None:
            import math
            return float(math.exp(float(avg_logprob)))

        return 0.0

    # ============================================================
    # Splitting (оставляем как есть)
    # ============================================================

    def _split_audio_into_speech_chunks(self, audio_path: str) -> List[_Chunk]:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("ffmpeg/ffprobe not found in PATH. Install them and add to PATH.")

        duration = self._probe_duration(audio_path)
        silences = self._detect_silences(audio_path)

        speech_ranges: List[Tuple[float, float]] = []
        cur = 0.0
        for s_start, s_end in silences:
            if s_start > cur:
                speech_ranges.append((cur, s_start))
            cur = max(cur, s_end)
        if cur < duration:
            speech_ranges.append((cur, duration))

        cleaned: List[Tuple[float, float]] = []
        for a, b in speech_ranges:
            a2 = max(0.0, a - self.cut_pad_sec)
            b2 = min(duration, b + self.cut_pad_sec)
            if b2 - a2 >= self.min_chunk_sec:
                cleaned.append((a2, b2))

        base_dir = os.path.dirname(os.path.abspath(audio_path))
        out_dir = os.path.join(base_dir, self.chunks_dir_name)
        os.makedirs(out_dir, exist_ok=True)

        chunks: List[_Chunk] = []
        for idx, (a, b) in enumerate(cleaned, start=1):
            out_path = os.path.join(out_dir, f"{os.path.basename(audio_path)}.chunk_{idx:04d}.wav")
            self._ffmpeg_cut(audio_path, out_path, a, b)
            chunks.append(_Chunk(idx=idx, src_start=a, src_end=b, path=out_path))

        return chunks

    def _probe_duration(self, audio_path: str) -> float:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", audio_path]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {p.stderr}")
        data = json.loads(p.stdout)
        return float(data["format"]["duration"])

    def _detect_silences(self, audio_path: str) -> List[Tuple[float, float]]:
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats",
            "-i", audio_path,
            "-af", f"silencedetect=noise={self.silence_db}dB:d={self.split_pause_sec}",
            "-f", "null", "-"
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = (p.stderr or "") + "\n" + (p.stdout or "")

        starts: List[float] = []
        ends: List[float] = []

        for line in out.splitlines():
            if "silence_start" in line:
                m = re.search(r"silence_start:\s*([0-9.]+)", line)
                if m:
                    starts.append(float(m.group(1)))
            if "silence_end" in line:
                m = re.search(r"silence_end:\s*([0-9.]+)", line)
                if m:
                    ends.append(float(m.group(1)))

        silences = []
        for s, e in zip(starts, ends):
            if e >= s:
                silences.append((s, e))
        silences.sort()
        return silences

    def _ffmpeg_cut(self, src: str, dst: str, start: float, end: float) -> None:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", src,
            "-ar", "16000",
            "-ac", "1",
            dst
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg cut failed: {p.stderr}")

    def _print_chunks_info(self, chunks: List[_Chunk]) -> None:
        total = sum(c.dur for c in chunks)
        logger.info(f"Speech chunks: {len(chunks)} | total speech ~ {total:.1f}s")
        print("\n========== CHUNKS ==========")
        for c in chunks:
            print(
                f"#{c.idx:04d}  start={c.src_start:8.2f}s  end={c.src_end:8.2f}s  "
                f"dur={c.dur:6.2f}s  file={os.path.basename(c.path)}"
            )
        print("========== /CHUNKS ==========\n")

    # ============================================================
    # helpers (твои)
    # ============================================================

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
            utts.append(
                Utterance(
                    role=cur_tokens[0].role,
                    start=float(start),
                    end=float(end),
                    text=text,
                    avg_prob=avg_prob,
                    words_count=len(cur_tokens),
                )
            )
            cur_tokens = []

        for w in words:
            if not cur_tokens:
                cur_tokens.append(w)
                continue
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
            lang = detect_lang(text)
            out.append(
                {
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
                    },
                }
            )
        return out
