from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Dict, List


def estimate_segment_duration(max_chunk_size_mb: float, bitrate_kbps: int) -> float:
    max_bytes = max_chunk_size_mb * 1024 * 1024
    bytes_per_second = (bitrate_kbps * 1000) / 8
    return max_bytes / bytes_per_second


def split_audio(
    audio_path: Path,
    output_dir: Path,
    duration_sec: float,
    max_chunk_size_mb: float,
    overlap_sec: float,
    bitrate_kbps: int,
) -> List[Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_duration = estimate_segment_duration(max_chunk_size_mb, bitrate_kbps)
    if segment_duration <= 0:
        raise ValueError("Invalid segment duration computed")
    step = max(segment_duration - overlap_sec, 1)
    total_segments = math.ceil(duration_sec / step)
    segments: List[Dict[str, object]] = []
    start = 0.0
    for index in range(total_segments):
        segment_path = output_dir / f"segment_{index:04d}.mp3"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            f"{start:.2f}",
            "-t",
            f"{segment_duration:.2f}",
            "-acodec",
            "copy",
            str(segment_path),
        ]
        subprocess.run(command, check=True, capture_output=True)
        segments.append(
            {
                "index": index,
                "start": start,
                "duration": segment_duration,
                "path": str(segment_path),
                "status": "pending",
                "output_json": None,
                "output_markdown": None,
                "handoff_path": None,
                "file_deleted": False,
                "error": None,
                "attempts_transient": 0,
                "attempts_semantic": 0,
            }
        )
        start += step
    return segments
