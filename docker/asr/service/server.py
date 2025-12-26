import os
import tempfile
import subprocess
from typing import List, Dict, Optional

import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

app = FastAPI(title="ASR Diarization Service", version="1.0")

# --- Lazy singletons ---
_DIAR_PIPELINE = None
_ASR_MODEL = None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_diar_pipeline():
    global _DIAR_PIPELINE
    if _DIAR_PIPELINE is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is not set in container env")
        # stable diarization model
        _DIAR_PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
        _DIAR_PIPELINE.to(torch.device(_device()))
    return _DIAR_PIPELINE


def get_asr_model(model_size: str):
    global _ASR_MODEL
    if _ASR_MODEL is None:
        compute_type = "float16" if _device() == "cuda" else "int8"
        _ASR_MODEL = WhisperModel(model_size, device=_device(), compute_type=compute_type)
    return _ASR_MODEL


def ffmpeg_to_wav16k_mono(src_path: str, dst_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        dst_path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr[:2000]}")


def diarize_segments(wav_path: str, num_speakers: int = 2) -> List[Dict]:
    pipeline = get_diar_pipeline()
    diar = pipeline(wav_path, num_speakers=num_speakers)

    segs = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        segs.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })
    segs.sort(key=lambda x: (x["start"], x["end"]))
    return segs


def merge_segments(segs: List[Dict], max_gap: float = 0.35, min_len: float = 0.6) -> List[Dict]:
    out = []
    cur = None
    for s in segs:
        if cur is None:
            cur = dict(s)
            continue
        same = s["speaker"] == cur["speaker"]
        gap = s["start"] - cur["end"]
        if same and gap <= max_gap:
            cur["end"] = max(cur["end"], s["end"])
        else:
            if (cur["end"] - cur["start"]) >= min_len:
                out.append(cur)
            cur = dict(s)
    if cur and (cur["end"] - cur["start"]) >= min_len:
        out.append(cur)
    return out


def transcribe_wav_segments(
    wav_path: str,
    segs: List[Dict],
    model_size: str = "small",
    language: Optional[str] = None,
) -> List[Dict]:
    model = get_asr_model(model_size)
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        raise RuntimeError(f"Expected 16kHz wav, got {sr}")

    results = []
    for s in segs:
        a = audio[int(s["start"] * sr): int(s["end"] * sr)]
        # faster-whisper expects numpy array; returns generator of segments
        fw_segs, info = model.transcribe(
            a,
            language=language,      # None -> auto
            vad_filter=False,       # VAD already handled by diarization
            beam_size=5,
        )
        text = " ".join([seg.text.strip() for seg in fw_segs]).strip()
        results.append({
            "start": s["start"],
            "end": s["end"],
            "speaker": s["speaker"],
            "text": text,
        })
    return results


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    num_speakers: int = Form(2),
    whisper_model: str = Form("small"),
    language: Optional[str] = Form(None),  # "en" / "ru" / None (auto)
):
    if num_speakers not in (1, 2, 3, 4):
        raise HTTPException(status_code=400, detail="num_speakers must be 1..4")

    suffix = os.path.splitext(file.filename or "")[1] or ".bin"

    with tempfile.TemporaryDirectory() as tmp:
        src_path = os.path.join(tmp, f"input{suffix}")
        wav_path = os.path.join(tmp, "audio_16k_mono.wav")

        content = await file.read()
        with open(src_path, "wb") as f:
            f.write(content)

        try:
            ffmpeg_to_wav16k_mono(src_path, wav_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        try:
            segs = diarize_segments(wav_path, num_speakers=num_speakers)
            segs = merge_segments(segs)
            utterances = transcribe_wav_segments(
                wav_path,
                segs,
                model_size=whisper_model,
                language=language,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return {
            "device": _device(),
            "num_speakers": num_speakers,
            "whisper_model": whisper_model,
            "language": language or "auto",
            "utterances": utterances,
        }


@app.get("/health")
def health():
    return {"ok": True, "device": _device()}
