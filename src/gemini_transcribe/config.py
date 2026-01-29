from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class JobConfig:
    job: Dict[str, Any]
    segmentation: Dict[str, Any]
    roles: Dict[str, Any]
    prompt: Dict[str, Any]
    execution: Dict[str, Any]
    storage: Dict[str, Any]
    output: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job": self.job,
            "segmentation": self.segmentation,
            "roles": self.roles,
            "prompt": self.prompt,
            "execution": self.execution,
            "storage": self.storage,
            "output": self.output,
        }


def default_config() -> JobConfig:
    return JobConfig(
        job={
            "id": None,
            "input": "path/to/input.mp4",
            "title": "",
        },
        segmentation={
            "max_chunk_size_mb": 20,
            "overlap_sec": 5,
            "audio_bitrate_kbps": 96,
            "audio_format": "mp3",
        },
        roles={
            "speaker_prefix": "Speaker",
        },
        prompt={
            "version": "v1",
        },
        execution={
            "model": "gemini-2.0-flash-001",
            "resume": True,
            "max_retries_transient": 3,
            "max_retries_semantic": 2,
        },
        storage={
            "work_dir": "outputs",
        },
        output={
            "write_legacy_txt": False,
            "legacy_output_path": None,
        },
    )


def load_job_config(path: Path) -> JobConfig:
    data = yaml.safe_load(path.read_text()) or {}
    defaults = default_config()

    def merge(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = default.copy()
        for key, value in (override or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    return JobConfig(
        job=merge(defaults.job, data.get("job", {})),
        segmentation=merge(defaults.segmentation, data.get("segmentation", {})),
        roles=merge(defaults.roles, data.get("roles", {})),
        prompt=merge(defaults.prompt, data.get("prompt", {})),
        execution=merge(defaults.execution, data.get("execution", {})),
        storage=merge(defaults.storage, data.get("storage", {})),
        output=merge(defaults.output, data.get("output", {})),
    )


def write_job_template(path: Path) -> None:
    template = {
        "job": {
            "id": "example-job",
            "input": "path/to/input.mp4",
            "title": "Example Interview",
        },
        "segmentation": {
            "max_chunk_size_mb": 20,
            "overlap_sec": 5,
            "audio_bitrate_kbps": 96,
            "audio_format": "mp3",
        },
        "roles": {
            "speaker_prefix": "Speaker",
        },
        "prompt": {
            "version": "v1",
        },
        "execution": {
            "model": "gemini-2.0-flash-001",
            "resume": True,
            "max_retries_transient": 3,
            "max_retries_semantic": 2,
        },
        "storage": {
            "work_dir": "outputs",
        },
    }
    path.write_text(
        "# Job definition for gemini-transcribe\n" + yaml.safe_dump(template, sort_keys=False)
    )


def write_effective_config(path: Path, config: JobConfig) -> None:
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
