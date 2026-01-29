from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from rich.console import Console
from rich.panel import Panel

from .config import load_job_config, write_job_template
from .gemini_client import GeminiClient, parse_json, retry_with_backoff
from .preprocess import FfmpegNotFoundError, prepare_audio
from .prompting import build_handoff_prompt, build_prompt, load_prompt_bundle
from .render import merge_segments, write_by_role, write_json, write_legacy_text, write_timeline
from .repair import build_repair_prompt
from .segmenter import split_audio
from .validate import response_schema, validate_segment_data
from .workspace import (
    ensure_state_exists,
    find_state_path,
    init_state,
    init_workspace,
    read_state,
    reset_segment_state,
    summarize_state,
    update_segment_state,
    write_state,
)

console = Console()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def templates_dir() -> Path:
    return repo_root() / "templates"


def prompts_template_dir() -> Path:
    return repo_root() / "prompts"


def copy_prompt_templates(destination: Path) -> None:
    source = prompts_template_dir()
    destination.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(f"Prompt templates missing at {source}")
    for path in source.rglob("*"):
        if path.is_file():
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_text(path.read_text())


def ensure_prompts(job_dir: Path) -> None:
    prompts_dir = job_dir / "prompts"
    if not prompts_dir.exists():
        copy_prompt_templates(prompts_dir)


def resolve_prompts_dir(job_path: Optional[Path], job_dir: Path) -> Path:
    if job_path is not None:
        candidate = job_path.parent / "prompts"
        if candidate.exists():
            return candidate
    prompts_dir = job_dir / "prompts"
    if prompts_dir.exists():
        return prompts_dir
    return prompts_template_dir()


def init_command(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_job_template(path)
    prompts_dir = path.parent / "prompts"
    copy_prompt_templates(prompts_dir)
    console.print(Panel(f"Initialized job template at {path}", style="green"))


def load_config_with_override(job_path: Path, model: Optional[str], prompt_version: Optional[str]):
    config = load_job_config(job_path)
    if model:
        config.execution["model"] = model
    if prompt_version:
        config.prompt["version"] = prompt_version
    return config


def build_segment_handoff(data: Dict[str, Any]) -> str:
    roles = ", ".join(role["role"] for role in data.get("roles", []))
    summary = data.get("summary", "")
    return f"Summary: {summary}\nRoles: {roles}"


def process_segment(
    client: GeminiClient,
    segment: Dict[str, Any],
    segment_index: int,
    total_segments: int,
    prompts_dir: Path,
    prompt_version: str,
    mime_type: str,
    max_retries_transient: int,
    max_retries_semantic: int,
    state: Dict[str, Any],
    stream: bool,
) -> Dict[str, Any]:
    bundle = load_prompt_bundle(prompts_dir, prompt_version)
    previous_handoff = None
    if segment_index > 0:
        previous_segment = state["segments"][segment_index - 1]
        if previous_segment.get("handoff_path"):
            previous_handoff = Path(previous_segment["handoff_path"]).read_text()
    prompt = build_prompt(bundle, segment_index, total_segments, previous_handoff)
    handoff_prompt = build_handoff_prompt(bundle, segment_index, total_segments)

    file_obj = retry_with_backoff(
        lambda: client.upload_file(segment["path"]),
        max_retries=max_retries_transient,
    )
    response_text = retry_with_backoff(
        lambda: client.generate_json(
            file_obj.uri,
            prompt,
            response_schema(),
            mime_type,
            stream=stream,
        ),
        max_retries=max_retries_transient,
    )
    data = None
    errors: List[str] = []
    raw_text = response_text
    for attempt in range(max_retries_semantic + 1):
        try:
            data = parse_json(raw_text)
        except json.JSONDecodeError as exc:  # noqa: PERF203
            errors.append(str(exc))
            data = None
        if data is not None:
            valid, validation_errors = validate_segment_data(data)
            if valid:
                break
            errors.extend(validation_errors)
            data = None
        if attempt < max_retries_semantic:
            repair_prompt = build_repair_prompt(raw_text, errors, response_schema())
            raw_text = client.repair_json(repair_prompt, response_schema())
        else:
            raise ValueError(f"Failed to validate JSON after repairs: {errors}")

    handoff_response = client.repair_json(handoff_prompt + "\n\n" + raw_text, response_schema())
    handoff_data = parse_json(handoff_response)
    handoff_summary = build_segment_handoff(handoff_data)

    client.delete_file(file_obj.name)
    return {"data": data, "handoff_summary": handoff_summary, "raw": raw_text}


def run_command(job_path: Path, resume: bool, prompt_version: Optional[str], model: Optional[str]) -> None:
    config = load_config_with_override(job_path, model, prompt_version)
    config.execution["resume"] = resume
    state_dir = Path(config.storage.get("work_dir", "outputs")) / (
        config.job.get("id") or job_path.stem
    )
    state_path = find_state_path(state_dir)
    if state_path.exists() and resume:
        state = read_state(state_path)
        job_dir = Path(state["job_dir"])
        effective_config = job_dir / "job.effective.yaml"
        if effective_config.exists():
            config = load_config_with_override(effective_config, model, prompt_version)
        console.print(Panel(f"Resuming job {state['job_id']}", style="blue"))
    else:
        workspace = init_workspace(config, job_path)
        job_dir = workspace.root
        console.print(Panel("Preparing audio", style="blue"))
        try:
            audio_path, duration = prepare_audio(
                Path(config.job["input"]),
                workspace.root,
                int(config.segmentation.get("audio_bitrate_kbps", 96)),
                str(config.segmentation.get("audio_format", "mp3")),
            )
        except FfmpegNotFoundError as exc:
            console.print(f"[red]{exc}[/]")
            raise
        console.print(Panel("Segmenting audio", style="blue"))
        segments = split_audio(
            audio_path,
            workspace.segments_dir,
            duration,
            float(config.segmentation.get("max_chunk_size_mb", 20)),
            float(config.segmentation.get("overlap_sec", 5)),
            int(config.segmentation.get("audio_bitrate_kbps", 96)),
        )
        state = init_state(workspace, segments, config)
        state["execution"] = config.execution
        write_state(workspace.state_path, state)
        state_path = workspace.state_path

    ensure_prompts(job_dir)
    prompts_dir = resolve_prompts_dir(job_path, job_dir)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set")
    client = GeminiClient(api_key, config.execution.get("model"))
    mime_type = "audio/mpeg"
    total_segments = len(state["segments"])

    for segment in state["segments"]:
        if segment.get("status") == "complete" and resume:
            continue
        index = segment["index"]
        state["current_segment"] = index
        write_state(state_path, state)
        console.print(Panel(f"Processing segment {index + 1}/{total_segments}", style="purple"))
        try:
            result = process_segment(
                client,
                segment,
                index,
                total_segments,
                prompts_dir,
                config.prompt.get("version", "v1"),
                mime_type,
                int(config.execution.get("max_retries_transient", 3)),
                int(config.execution.get("max_retries_semantic", 2)),
                state,
                stream=True,
            )
            segment_json_path = job_dir / "segments" / f"segment_{index:04d}.json"
            segment_md_path = job_dir / "segments" / f"segment_{index:04d}.md"
            handoff_path = job_dir / "segments" / f"segment_{index:04d}_handoff.txt"
            write_json(segment_json_path, result["data"])
            write_timeline(segment_md_path, [result["data"]])
            handoff_path.write_text(result["handoff_summary"])
            update_segment_state(
                state,
                index,
                {
                    "status": "complete",
                    "output_json": str(segment_json_path),
                    "output_markdown": str(segment_md_path),
                    "handoff_path": str(handoff_path),
                    "file_deleted": True,
                    "error": None,
                },
            )
            state["last_error"] = None
        except Exception as exc:  # noqa: BLE001
            update_segment_state(
                state,
                index,
                {
                    "status": "failed",
                    "error": str(exc),
                },
            )
            state["last_error"] = str(exc)
            write_state(state_path, state)
            raise
        write_state(state_path, state)

    merge_outputs(job_dir, state, config)
    state["status"] = "completed"
    state["current_segment"] = None
    write_state(state_path, state)
    console.print(Panel(f"Job completed. Outputs in {job_dir / 'final'}", style="green"))


def merge_outputs(job_dir: Path, state: Dict[str, Any], config) -> None:
    segments_data = []
    for segment in state["segments"]:
        if segment.get("output_json"):
            segments_data.append(json.loads(Path(segment["output_json"]).read_text()))
    final_dir = job_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    merged = merge_segments(segments_data)
    merged_path = final_dir / "merged.json"
    write_json(merged_path, merged)
    timeline_path = final_dir / "timeline.md"
    by_role_path = final_dir / "by_role.md"
    write_timeline(timeline_path, segments_data)
    write_by_role(by_role_path, segments_data)
    state["final_outputs"] = {
        "merged": str(merged_path),
        "timeline": str(timeline_path),
        "by_role": str(by_role_path),
    }
    if config.output.get("write_legacy_txt") and config.output.get("legacy_output_path"):
        write_legacy_text(Path(config.output["legacy_output_path"]), segments_data)


def status_command(job_dir: Path) -> None:
    state = ensure_state_exists(find_state_path(job_dir))
    summary = summarize_state(state)
    console.print(Panel(json.dumps(summary, indent=2), style="cyan"))


def rerun_command(job_dir: Path, segment_index: int, prompt_version: Optional[str], model: Optional[str]) -> None:
    state_path = find_state_path(job_dir)
    state = ensure_state_exists(state_path)
    effective_config_path = Path(state["job_dir"]) / "job.effective.yaml"
    config = load_config_with_override(effective_config_path, model, prompt_version)
    reset_segment_state(state, segment_index)
    segment = state["segments"][segment_index]
    for key in ["output_json", "output_markdown", "handoff_path"]:
        if segment.get(key):
            path = Path(segment[key])
            if path.exists():
                path.unlink()
    write_state(state_path, state)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set")
    client = GeminiClient(api_key, config.execution.get("model"))
    total_segments = len(state["segments"])
    result = process_segment(
        client,
        segment,
        segment_index,
        total_segments,
        resolve_prompts_dir(None, Path(state["job_dir"])),
        config.prompt.get("version", "v1"),
        "audio/mpeg",
        int(config.execution.get("max_retries_transient", 3)),
        int(config.execution.get("max_retries_semantic", 2)),
        state,
        stream=True,
    )
    segment_json_path = Path(state["job_dir"]) / "segments" / f"segment_{segment_index:04d}.json"
    segment_md_path = Path(state["job_dir"]) / "segments" / f"segment_{segment_index:04d}.md"
    handoff_path = Path(state["job_dir"]) / "segments" / f"segment_{segment_index:04d}_handoff.txt"
    write_json(segment_json_path, result["data"])
    write_timeline(segment_md_path, [result["data"]])
    handoff_path.write_text(result["handoff_summary"])
    update_segment_state(
        state,
        segment_index,
        {
            "status": "complete",
            "output_json": str(segment_json_path),
            "output_markdown": str(segment_md_path),
            "handoff_path": str(handoff_path),
            "file_deleted": True,
            "error": None,
        },
    )
    write_state(state_path, state)
    merge_outputs(Path(state["job_dir"]), state, config)
    console.print(Panel(f"Segment {segment_index} rerun complete", style="green"))


def prompt_list_command(job_dir: Optional[Path]) -> None:
    base = job_dir / "prompts" if job_dir else prompts_template_dir()
    if not base.exists():
        raise FileNotFoundError(f"Prompts directory not found at {base}")
    versions = [p.name for p in base.iterdir() if p.is_dir()]
    console.print("\n".join(sorted(versions)))


def prompt_show_command(job_dir: Optional[Path], version: str) -> None:
    base = job_dir / "prompts" if job_dir else prompts_template_dir()
    for path in sorted((base / version).glob("*.txt")):
        console.print(Panel(path.read_text(), title=path.name, style="green"))


def legacy_args_to_job(args: argparse.Namespace) -> Path:
    job_path = Path("job_legacy.yaml")
    if job_path.exists():
        job_path.unlink()
    template = {
        "job": {"id": "legacy", "input": args.input},
        "segmentation": {
            "max_chunk_size_mb": 2000,
            "overlap_sec": 0,
            "audio_bitrate_kbps": 96,
            "audio_format": "mp3",
        },
        "prompt": {"version": "v1"},
        "execution": {"model": args.model or "gemini-2.0-flash-001", "resume": False},
        "storage": {"work_dir": "outputs"},
        "output": {"write_legacy_txt": bool(args.output), "legacy_output_path": args.output},
    }
    job_path.write_text(yaml.safe_dump(template, sort_keys=False))
    return job_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemini transcribe workflow")
    parser.add_argument("--input", type=str, help="Legacy input path")
    parser.add_argument("--model", type=str, help="Legacy model override")
    parser.add_argument("--output", type=str, help="Legacy output path")

    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Create a job.yaml template")
    init_parser.add_argument("--path", type=Path, default=Path("job.yaml"))

    run_parser = subparsers.add_parser("run", help="Run a job")
    run_parser.add_argument("job_path", type=Path)
    run_parser.add_argument("--resume", action="store_true", default=True)
    run_parser.add_argument("--no-resume", dest="resume", action="store_false")
    run_parser.add_argument("--prompt", dest="prompt_version", type=str)
    run_parser.add_argument("--model", dest="model", type=str)

    status_parser = subparsers.add_parser("status", help="Show job status")
    status_parser.add_argument("job_dir", type=Path)

    rerun_parser = subparsers.add_parser("rerun", help="Rerun a segment")
    rerun_parser.add_argument("--job", dest="job_dir", type=Path, required=True)
    rerun_parser.add_argument("--segment", type=int, required=True)
    rerun_parser.add_argument("--prompt", dest="prompt_version", type=str)
    rerun_parser.add_argument("--model", dest="model", type=str)

    prompt_parser = subparsers.add_parser("prompt", help="Prompt utilities")
    prompt_subparsers = prompt_parser.add_subparsers(dest="prompt_command")
    prompt_list = prompt_subparsers.add_parser("list", help="List prompt versions")
    prompt_list.add_argument("--job", dest="job_dir", type=Path)
    prompt_show = prompt_subparsers.add_parser("show", help="Show prompt version")
    prompt_show.add_argument("version", type=str)
    prompt_show.add_argument("--job", dest="job_dir", type=Path)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None and args.input:
        job_path = legacy_args_to_job(args)
        run_command(job_path, resume=False, prompt_version=None, model=args.model)
        return

    if args.command == "init":
        init_command(args.path)
    elif args.command == "run":
        run_command(args.job_path, resume=args.resume, prompt_version=args.prompt_version, model=args.model)
    elif args.command == "status":
        status_command(args.job_dir)
    elif args.command == "rerun":
        rerun_command(args.job_dir, args.segment, args.prompt_version, args.model)
    elif args.command == "prompt" and args.prompt_command == "list":
        prompt_list_command(args.job_dir)
    elif args.command == "prompt" and args.prompt_command == "show":
        prompt_show_command(args.job_dir, args.version)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
