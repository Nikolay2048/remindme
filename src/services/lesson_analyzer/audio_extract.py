from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class ExtractedTracks:
    student_wav: str
    teacher_wav: str
    mixed_wav: str

class AudioExtractor:
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

    def extract_three_tracks(self, *, video_path: str, out_dir: str) -> ExtractedTracks:
        # TODO: В телеге передается только один трек
        student = f"{out_dir}/student.wav"
        teacher = f"{out_dir}/teacher.wav"
        mixed = f"{out_dir}/mixed.wav"

        # by your convention:
        self._extract_track(video_path, 0, student)
        self._extract_track(video_path, 1, teacher)
        self._extract_track(video_path, 2, mixed)

        return ExtractedTracks(student_wav=student, teacher_wav=teacher, mixed_wav=mixed)
