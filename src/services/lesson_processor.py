import os
import json
import uuid
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter

from faster_whisper import WhisperModel


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: Optional[float]
    pause_ms: Optional[int]
    role: str


class LessonProcessor:
    def __init__(
        self,
        *,
        whisper_model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
        low_prob: float = 0.55,
        high_prob: float = 0.75,
        long_pause_ms: int = 500,
        contexts_keep: int = 5000,  # в MVP можно много, потом резать
    ):
        self.model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
        self.language = language

        self.low_prob = low_prob
        self.high_prob = high_prob
        self.long_pause_ms = long_pause_ms
        self.contexts_keep = contexts_keep

        # MVP набор fillers (позже расширим фразами/регэкспами)
        self.fillers = {
            "um", "uh", "erm", "hmm", "like", "well", "actually", "basically"
        }
        self.filler_phrases = {("you", "know"): "you know"}

    # ----------------------------
    # Public API
    # ----------------------------

    def analyze_video(self, video_path: str, out_dir: str = "_lesson") -> Dict[str, Any]:
        lesson_id = str(uuid.uuid4())
        print("start processing", video_path)
        os.makedirs(out_dir, exist_ok=True)
        lesson_dir = os.path.join(out_dir, f"lesson_{lesson_id}")
        os.makedirs(lesson_dir, exist_ok=True)

        # 1) Extract 3 tracks
        tracks = self._extract_three_tracks(video_path, lesson_dir)

        # 2) Transcribe teacher + student (mixed можно пока не анализировать)
        teacher_words, teacher_text = self._transcribe(tracks["teacher"], role="teacher")
        student_words, student_text = self._transcribe(tracks["student"], role="student")

        # 3) Analyze
        report = self._build_report(
            lesson_id=lesson_id,
            teacher_words=teacher_words,
            student_words=student_words,
            teacher_text_preview=teacher_text,
            student_text_preview=student_text,
        )

        # 4) Save JSON
        report_path = os.path.join(lesson_dir, "lesson_report1.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        report["paths"] = {**tracks, "report_json": report_path}
        return report

    # ----------------------------
    # Audio extraction
    # ----------------------------

    def _run(self, cmd: List[str]) -> None:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n\n{r.stderr}")

    def _extract_track(self, video: str, index: int, out_wav: str) -> None:
        self._run([
            "ffmpeg", "-y",
            "-i", video,
            "-map", f"0:a:{index}",
            "-ac", "1", "-ar", "16000",
            "-vn",
            out_wav
        ])

    def _extract_three_tracks(self, video: str, out_dir: str) -> Dict[str, str]:
        paths = {
            "teacher": os.path.join(out_dir, "teacher.wav"),
            "student": os.path.join(out_dir, "student.wav"),
            "mixed": os.path.join(out_dir, "mixed.wav"),
        }
        self._extract_track(video, 0, paths["student"])
        self._extract_track(video, 1, paths["teacher"])
        self._extract_track(video, 2, paths["mixed"])
        return paths

    # ----------------------------
    # ASR
    # ----------------------------

    def _transcribe(self, audio_path: str, role: str) -> (List[Word], str):
        segments, _info = self.model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        words: List[Word] = []
        text_parts: List[str] = []
        prev_end: Optional[float] = None

        for seg in segments:
            if seg.text:
                text_parts.append(seg.text.strip())

            for w in (seg.words or []):
                token = (w.word or "").strip().lower()
                if not token:
                    continue

                pause_ms = None
                if prev_end is not None and w.start is not None:
                    pause_ms = int(max(0.0, (w.start - prev_end) * 1000))
                prev_end = float(w.end) if w.end is not None else prev_end

                prob = None
                if hasattr(w, "probability") and w.probability is not None:
                    prob = float(w.probability)

                words.append(Word(
                    text=token,
                    start=float(w.start or 0.0),
                    end=float(w.end or 0.0),
                    prob=prob,
                    pause_ms=pause_ms,
                    role=role,
                ))

        return words, " ".join(text_parts).strip()

    # ----------------------------
    # Core analysis
    # ----------------------------

    def _normalize(self, t: str) -> str:
        # remove punctuation at edges, keep apostrophes inside
        t = t.strip().lower()
        t = t.strip(".,!?;:\"()[]{}<>")
        return t

    def _is_word(self, t: str) -> bool:
        t = self._normalize(t)
        return t.isalpha() or ("'" in t and t.replace("'", "").isalpha())

    def _detect_filler_phrase_positions(self, tokens: List[str]) -> set:
        """
        Marks indices that belong to filler phrases like ("you","know").
        """
        marked = set()
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in self.filler_phrases:
                marked.add(i)
                marked.add(i + 1)
        return marked

    def _build_report(
        self,
        *,
        lesson_id: str,
        teacher_words: List[Word],
        student_words: List[Word],
        teacher_text_preview: str,
        student_text_preview: str,
    ) -> Dict[str, Any]:
        all_words = teacher_words + student_words

        # talk time (приблизительно суммой длительностей слов)
        teacher_time = sum(max(0.0, w.end - w.start) for w in teacher_words)
        student_time = sum(max(0.0, w.end - w.start) for w in student_words)
        total_time = max(1e-6, teacher_time + student_time)

        # Build token streams per role for phrase detection
        teacher_tokens = [self._normalize(w.text) for w in teacher_words]
        student_tokens = [self._normalize(w.text) for w in student_words]
        teacher_filler_pos = self._detect_filler_phrase_positions(teacher_tokens)
        student_filler_pos = self._detect_filler_phrase_positions(student_tokens)

        # Frequency of words (student-only is often what we want for “your vocab”)
        student_word_freq = Counter(
            self._normalize(w.text) for w in student_words if self._is_word(w.text)
        )
        teacher_word_freq = Counter(
            self._normalize(w.text) for w in teacher_words if self._is_word(w.text)
        )

        contexts: List[Dict[str, Any]] = []
        issues: List[Dict[str, Any]] = []

        vocab_uncertain = []
        vocab_confident = []
        filler_hits = Counter()

        def process_word_list(words: List[Word], filler_pos: set):
            tokens_norm = [self._normalize(w.text) for w in words]
            for idx, w in enumerate(words):
                t = self._normalize(w.text)
                if not self._is_word(t):
                    continue

                flags = []

                # fillers
                is_filler = (t in self.fillers) or (idx in filler_pos)
                if is_filler:
                    flags.append("filler")
                    filler_hits[t] += 1

                # uncertainty
                low_prob = (w.prob is not None and w.prob < self.low_prob)
                long_pause = (w.pause_ms is not None and w.pause_ms >= self.long_pause_ms)

                if low_prob:
                    flags.append("low_confidence")
                if long_pause:
                    flags.append("pause_before")

                # context snippet window
                left = max(0, idx - 6)
                right = min(len(words), idx + 7)
                snippet = " ".join(words[i].text for i in range(left, right)).strip()

                ctx = {
                    "role": w.role,
                    "word": t,
                    "start_sec": w.start,
                    "end_sec": w.end,
                    "confidence": w.prob,
                    "pause_before_ms": w.pause_ms,
                    "flags": flags,
                    "context": snippet,
                }
                contexts.append(ctx)

                uncertain = (low_prob or long_pause) and not is_filler
                if uncertain:
                    vocab_uncertain.append(ctx)
                    issues.append({
                        "type": "uncertainty",
                        "role": w.role,
                        "word": t,
                        "start_sec": w.start,
                        "end_sec": w.end,
                        "confidence": w.prob,
                        "pause_before_ms": w.pause_ms,
                        "context": snippet,
                        "signals": [f for f in flags if f != "filler"],
                    })

                confident = (
                    w.role == "student"
                    and (w.prob is None or w.prob >= self.high_prob)
                    and (w.pause_ms is None or w.pause_ms < self.long_pause_ms)
                    and not is_filler
                )
                if confident:
                    vocab_confident.append(ctx)

        process_word_list(teacher_words, teacher_filler_pos)
        process_word_list(student_words, student_filler_pos)

        # New words (MVP): слова, которые student сказал, но teacher не говорил (в пределах урока)
        # (Это грубо, но для MVP даёт “личный” словарь ученика)
        new_words = []
        for w, c in student_word_freq.items():
            if c >= 1 and teacher_word_freq.get(w, 0) == 0 and len(w) >= 2:
                new_words.append({"word": w, "student_count": c})

        # Confident words: усилим условием “>=2 раза”
        confident_words = []
        for item in vocab_confident:
            w = item["word"]
            if student_word_freq.get(w, 0) >= 2:
                confident_words.append(item)

        # Student-centeredness (очень простой скор для MVP)
        student_talk_ratio = student_time / total_time
        avg_student_turn_len = len(student_words) / max(1, len(set([w.start for w in student_words])))
        student_centered_score = round(
            0.7 * student_talk_ratio + 0.3 * min(1.0, student_word_freq.total() / max(1, teacher_word_freq.total())),
            3
        )

        report = {
            "lesson_id": lesson_id,
            "lesson_meta": {
                "teacher_words": len(teacher_words),
                "student_words": len(student_words),
                "teacher_text_preview": teacher_text_preview,
                "student_text_preview": student_text_preview,
            },
            "talk_metrics": {
                "teacher_time_sec": round(teacher_time, 2),
                "student_time_sec": round(student_time, 2),
                "student_talk_ratio": round(student_talk_ratio, 3),
                "student_centered_score_mvp": student_centered_score,
            },
            "vocabulary": {
                "new": sorted(new_words, key=lambda x: (-x["student_count"], x["word"]))[:200],
                "uncertain": vocab_uncertain[:200],
                "confident": confident_words[:200],
                "fillers": [{"token": k, "count": v} for k, v in filler_hits.most_common(50)],
            },
            "issues": issues[:500],
            "contexts": contexts[: self.contexts_keep],
        }
        return report


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    print("starts")
    analyzer = LessonProcessor(whisper_model_size="small")
    report = analyzer.analyze_video("../../data/lessons/Arina_lesson_2.mp4", out_dir="_lesson")
    print("Saved report:", report["paths"]["report_json"])
