from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple


class FfmpegNotFoundError(RuntimeError):
    pass


def ensure_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        subprocess.run(["ffprobe", "-version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise FfmpegNotFoundError("ffmpeg/ffprobe is required but not found") from exc


def probe_duration(input_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def convert_to_audio(
    input_path: Path,
    output_path: Path,
    bitrate_kbps: int,
    audio_format: str,
) -> Path:
    ensure_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "libmp3lame" if audio_format == "mp3" else "pcm_s16le",
        "-b:a",
        f"{bitrate_kbps}k",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output_path


def derive_audio_path(input_path: Path, audio_format: str) -> Path:
    suffix = ".mp3" if audio_format == "mp3" else ".wav"
    return input_path.with_suffix(suffix)


def prepare_audio(
    input_path: Path,
    work_dir: Path,
    bitrate_kbps: int,
    audio_format: str,
) -> Tuple[Path, float]:
    audio_path = work_dir / f"source_audio.{audio_format}"
    audio_path = convert_to_audio(input_path, audio_path, bitrate_kbps, audio_format)
    duration = probe_duration(audio_path)
    return audio_path, duration
