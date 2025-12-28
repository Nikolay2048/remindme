from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import requests


@dataclass(frozen=True)
class Utterance:
    speaker: str   # SPEAKER_00 / SPEAKER_01
    start: float
    end: float
    text: str


def transcribe_two_speakers(
    audio_path: str,
    *,
    asr_url: Optional[str] = None,
    language: str,
    timeout_sec: int = 60 * 60,
) -> List[Utterance]:
    """
    Вызов WhisperX ASR Service и получение реплик с ролями говорящих.

    Ожидается сервис:
      POST {ASR_URL}/asr
      multipart: audio_file=<file>, language=<str>
    """
    base = (asr_url or os.getenv("ASR_URL", "http://localhost:9000")).rstrip("/")
    url = f"{base}/asr"

    with open(audio_path, "rb") as f:
        r = requests.post(
            url,
            files={"audio_file": (os.path.basename(audio_path), f)},
            data={"language": language},
            timeout=timeout_sec,
        )

    r.raise_for_status()
    payload = r.json()
    print(payload)
    segs = payload.get("segments")
    if not isinstance(segs, list) or not segs:
        raise ValueError("ASR service response does not contain non-empty 'segments' list.")

    utterances = _segments_to_utterances(segs)
    utterances = _map_to_two_speakers(utterances)      # -> SPEAKER_00/01
    utterances = _merge_adjacent(utterances, gap=0.25) # склеиваем подряд идущие
    return utterances


def _segments_to_utterances(segments: List[Dict[str, Any]]) -> List[Utterance]:
    out: List[Utterance] = []
    for s in segments:
        if not isinstance(s, dict):
            continue
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        spk = str(s.get("speaker", "SPEAKER_00")).strip()
        out.append(
            Utterance(
                speaker=_norm_label(spk),
                start=float(s.get("start", 0.0) or 0.0),
                end=float(s.get("end", 0.0) or 0.0),
                text=text,
            )
        )
    if not out:
        raise ValueError("No textual segments found in 'segments'.")
    return out


def _norm_label(label: str) -> str:
    x = label.upper().replace(" ", "_")
    x = x.replace("SPEAKER__", "SPEAKER_")
    if x.startswith("SPEAKER_"):
        return x
    if x.startswith("SPEAKER"):
        # SPEAKER0 -> SPEAKER_0, SPEAKER01 -> SPEAKER_01 (оставим как есть дальше замапим)
        tail = x.replace("SPEAKER", "")
        tail = tail if tail else "0"
        return f"SPEAKER_{tail}"
    return x


def _map_to_two_speakers(utts: List[Utterance]) -> List[Utterance]:
    """
    Берём первые два уникальных спикера, которые встретились в данных,
    и мапим их на SPEAKER_00 / SPEAKER_01.
    Остальных (если вдруг появились) — прижимаем к ближайшему из двух по умолчанию.
    """
    uniq: List[str] = []
    for u in utts:
        if u.speaker not in uniq:
            uniq.append(u.speaker)
        if len(uniq) == 2:
            break

    if len(uniq) == 1:
        mapping = {uniq[0]: "SPEAKER_00"}
    else:
        mapping = {uniq[0]: "SPEAKER_00", uniq[1]: "SPEAKER_01"}

    out: List[Utterance] = []
    for u in utts:
        sp = mapping.get(u.speaker)
        if sp is None:
            # если вдруг третий спикер — прижмём к 00
            sp = "SPEAKER_00"
        out.append(Utterance(speaker=sp, start=u.start, end=u.end, text=u.text))
    return out


def _merge_adjacent(utts: List[Utterance], *, gap: float = 0.25) -> List[Utterance]:
    """
    Склеиваем соседние сегменты одного и того же говорящего,
    если пауза между ними <= gap.
    """
    if not utts:
        return utts

    merged: List[Utterance] = [utts[0]]
    for u in utts[1:]:
        last = merged[-1]
        if u.speaker == last.speaker and (u.start - last.end) <= gap:
            merged[-1] = Utterance(
                speaker=last.speaker,
                start=last.start,
                end=max(last.end, u.end),
                text=(last.text + " " + u.text).strip(),
            )
        else:
            merged.append(u)
    return merged


# пример:
res = transcribe_two_speakers(r"video_lesson.mp4", language="auto")
for u in res:
    print(f"[{u.speaker}] {u.start:.2f}-{u.end:.2f}: {u.text}")
