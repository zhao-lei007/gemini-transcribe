from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import JobConfig, write_effective_config


@dataclass
class Workspace:
    job_id: str
    root: Path
    state_path: Path
    segments_dir: Path
    final_dir: Path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "root": str(self.root),
            "state_path": str(self.state_path),
            "segments_dir": str(self.segments_dir),
            "final_dir": str(self.final_dir),
        }


def resolve_job_id(config: JobConfig, job_path: Path) -> str:
    if config.job.get("id"):
        return str(config.job["id"])
    return job_path.stem or "job"


def init_workspace(config: JobConfig, job_path: Path) -> Workspace:
    job_id = resolve_job_id(config, job_path)
    work_dir = Path(config.storage.get("work_dir", "outputs"))
    root = work_dir / job_id
    segments_dir = root / "segments"
    final_dir = root / "final"
    root.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    effective_config_path = root / "job.effective.yaml"
    write_effective_config(effective_config_path, config)
    return Workspace(
        job_id=job_id,
        root=root,
        state_path=state_path,
        segments_dir=segments_dir,
        final_dir=final_dir,
    )


def load_workspace_from_state(state_path: Path) -> Workspace:
    data = json.loads(state_path.read_text())
    root = Path(data["job_dir"])
    return Workspace(
        job_id=data["job_id"],
        root=root,
        state_path=state_path,
        segments_dir=root / "segments",
        final_dir=root / "final",
    )


def init_state(workspace: Workspace, segments: List[Dict[str, Any]], config: JobConfig) -> Dict[str, Any]:
    return {
        "job_id": workspace.job_id,
        "job_dir": str(workspace.root),
        "status": "initialized",
        "model": config.execution.get("model"),
        "prompt_version": config.prompt.get("version"),
        "created_at": datetime.utcnow().isoformat(),
        "current_segment": None,
        "last_error": None,
        "segments": segments,
        "final_outputs": {},
    }


def read_state(state_path: Path) -> Dict[str, Any]:
    return json.loads(state_path.read_text())


def write_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.write_text(json.dumps(state, indent=2))


def update_segment_state(state: Dict[str, Any], index: int, updates: Dict[str, Any]) -> None:
    for segment in state["segments"]:
        if segment["index"] == index:
            segment.update(updates)
            return
    raise ValueError(f"Segment {index} not found in state")


def reset_segment_state(state: Dict[str, Any], index: int) -> None:
    for segment in state["segments"]:
        if segment["index"] == index:
            segment.update(
                {
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
            return
    raise ValueError(f"Segment {index} not found in state")


def find_state_path(job_dir: Path) -> Path:
    if job_dir.is_file():
        return job_dir
    return job_dir / "state.json"


def ensure_state_exists(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        raise FileNotFoundError(f"state.json not found at {state_path}")
    return read_state(state_path)


def summarize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    total = len(state.get("segments", []))
    completed = sum(1 for s in state.get("segments", []) if s.get("status") == "complete")
    current = state.get("current_segment")
    return {
        "total_segments": total,
        "completed_segments": completed,
        "current_segment": current,
        "last_error": state.get("last_error"),
        "status": state.get("status"),
        "final_outputs": state.get("final_outputs"),
    }


def list_segment_outputs(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return state.get("segments", [])
