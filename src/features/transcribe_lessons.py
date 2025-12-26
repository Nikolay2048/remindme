import os
import re
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import soundfile as sf
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

load_dotenv()


# ---------------------------
# Config / cache
# ---------------------------
_DIAR: Optional[Pipeline] = None
_ASR: Dict[str, WhisperModel] = {}


def _device(prefer: str = "cuda") -> str:
    return "cuda" if prefer == "cuda" and torch.cuda.is_available() else "cpu"


def _ffmpeg_to_wav16k_mono(src_path: str, dst_path: str) -> None:
    cmd = ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", "-vn", dst_path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{p.stderr[:4000]}")


def _load_pipeline(model_id: str, token: str) -> Pipeline:
    try:
        return Pipeline.from_pretrained(model_id, use_auth_token=token)
    except TypeError:
        return Pipeline.from_pretrained(model_id, token=token)


def _get_diar(hf_token: Optional[str], device: str) -> Pipeline:
    global _DIAR
    if _DIAR is None:
        token = hf_token or os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is not set. Put it into .env or pass hf_token=...")
        _DIAR = _load_pipeline("pyannote/speaker-diarization-3.1", token)
        _DIAR.to(torch.device(device))
    return _DIAR


def _get_asr(model_size: str, device: str) -> WhisperModel:
    key = f"{model_size}:{device}"
    if key not in _ASR:
        compute_type = "float16" if device == "cuda" else "int8"
        _ASR[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _ASR[key]


def _wav_to_file_dict(wav_path: str) -> Dict:
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        raise RuntimeError(f"Expected 16kHz wav, got {sr}")

    # pyannote expects (channels, time)
    if audio.ndim == 1:
        audio = audio[None, :]
    else:
        audio = audio.T

    return {"waveform": torch.from_numpy(audio).float(), "sample_rate": sr}


# ---------------------------
# RTTM parsing (stable)
# ---------------------------
_RTTM_RE = re.compile(
    r"^SPEAKER\s+\S+\s+1\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+<NA>\s+<NA>\s+(\S+)\s+<NA>\s+<NA>\s*$"
)


@dataclass
class Seg:
    start: float
    end: float
    speaker: str


def _read_rttm(rttm_path: str) -> List[Seg]:
    segs: List[Seg] = []
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _RTTM_RE.match(line)
            if not m:
                continue
            start = float(m.group(1))
            dur = float(m.group(2))
            speaker = m.group(3)
            segs.append(Seg(start=start, end=start + dur, speaker=speaker))
    segs.sort(key=lambda s: (s.start, s.end))
    return segs


def _merge(segs: List[Seg], max_gap: float = 0.6, min_len: float = 1.0) -> List[Seg]:
    if not segs:
        return []
    out: List[Seg] = []
    cur = segs[0]

    for s in segs[1:]:
        if s.speaker == cur.speaker and (s.start - cur.end) <= max_gap:
            cur = Seg(cur.start, max(cur.end, s.end), cur.speaker)
        else:
            if (cur.end - cur.start) >= min_len:
                out.append(cur)
            cur = s

    if (cur.end - cur.start) >= min_len:
        out.append(cur)
    return out


def _transcribe_segments(wav_path: str, segs: List[Seg], model_size: str, device: str,
                         language: Optional[str], beam_size: int) -> List[Dict]:
    model = _get_asr(model_size, device)
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        raise RuntimeError(f"Expected 16kHz wav, got {sr}")

    out: List[Dict] = []
    for s in segs:
        chunk = audio[int(s.start * sr): int(s.end * sr)]
        fw_segs, _ = model.transcribe(
            chunk,
            language=language,      # None -> auto (лучше для RU+EN)
            vad_filter=False,
            beam_size=beam_size,
        )
        text = " ".join(x.text.strip() for x in fw_segs).strip()
        out.append({"start": s.start, "end": s.end, "speaker": s.speaker, "text": text})
    return out


# ---------------------------
# Public function
# ---------------------------
def transcribe_lesson_two_speakers(
    media_path: str,
    *,
    hf_token: Optional[str] = None,
    num_speakers: int = 2,
    whisper_model: str = "small",
    language: Optional[str] = None,       # None=auto; "ru"/"en"
    merge_max_gap: float = 0.6,
    merge_min_len: float = 1.0,
    beam_size: int = 5,
) -> Dict:
    device = "cpu"

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "audio.wav")
        rttm_path = os.path.join(tmp, "diarization.rttm")

        _ffmpeg_to_wav16k_mono(media_path, wav_path)

        diar = _get_diar(hf_token, device)
        file_dict = _wav_to_file_dict(wav_path)

        diar_out = diar(file_dict, num_speakers=num_speakers)

        # КЛЮЧ: конвертируем в RTTM (стабильно для pyannote 4.x)
        # diar_out может быть DiarizeOutput — write_rttm у него есть практически всегда
        if not hasattr(diar_out, "write_rttm"):
            raise RuntimeError(f"Unexpected diarization output: {type(diar_out)} (no write_rttm)")

        with open(rttm_path, "w", encoding="utf-8") as f:
            diar_out.write_rttm(f)

        segs = _read_rttm(rttm_path)
        segs = _merge(segs, max_gap=merge_max_gap, min_len=merge_min_len)

        utterances = _transcribe_segments(
            wav_path, segs, model_size=whisper_model, device=device, language=language, beam_size=beam_size
        )

        return {
            "device": device,
            "num_speakers": num_speakers,
            "whisper_model": whisper_model,
            "language": language or "auto",
            "utterances": utterances,
        }


if __name__ == "__main__":
    video_path = r"C:\Users\Admin\PycharmProjects\ReMindMe\tests\video.mp4"
    res = transcribe_lesson_two_speakers(
        video_path,
        num_speakers=2,
        whisper_model="small",
        language=None,
    )
    for u in res["utterances"][:30]:
        print(f'[{u["start"]:.2f}-{u["end"]:.2f}] {u["speaker"]}: {u["text"]}')
