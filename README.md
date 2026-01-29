# gemini-transcribe

A local, job-driven workflow for transcribing MP4 audio with Gemini, with resumable state, prompt versioning, and structured outputs.

## Features

- `job.yaml`-driven workflow with resume support
- MP4 → audio stabilization → size-based segmentation → serial transcription
- Structured JSON output + timeline and role views
- Prompt versioning with per-job prompt bundles
- Rerun single segments for prompt comparison

## Prerequisites

- Python 3.12+
- `ffmpeg` and `ffprobe` available on your PATH
- Gemini Developer API key set as `GEMINI_API_KEY`

```bash
export GEMINI_API_KEY="your-api-key"
```

## Installation

```bash
pip install -e .
```

## Usage

<details>
<summary>中文（点击切换）</summary>

### 快速开始

```bash
transcribe init --path job.yaml
# 编辑 job.yaml，设置输入 MP4 路径和 job id
transcribe run job.yaml
```

### 常用命令

```bash
transcribe status outputs/<job_id>
transcribe rerun --job outputs/<job_id> --segment 2 --prompt v1
```

> 提示：需确保环境变量 `GEMINI_API_KEY` 已设置，且本机安装 `ffmpeg`/`ffprobe`。

</details>

<details>
<summary>English (click to switch)</summary>

### Quick start

```bash
transcribe init --path job.yaml
# Edit job.yaml to point at your input MP4 and job id
transcribe run job.yaml
```

### Common commands

```bash
transcribe status outputs/<job_id>
transcribe rerun --job outputs/<job_id> --segment 2 --prompt v1
```

> Tip: ensure `GEMINI_API_KEY` is set and `ffmpeg`/`ffprobe` are installed.

</details>

### Initialize a job

```bash
transcribe init --path job.yaml
```

This creates a `job.yaml` template and a `prompts/v1` directory.

### Run a job (with resume)

```bash
transcribe run job.yaml
```

### Check status

```bash
transcribe status outputs/<job_id>
```

### Rerun a segment

```bash
transcribe rerun --job outputs/<job_id> --segment 2 --prompt v1
```

### End-to-end example

```bash
transcribe init --path job.yaml
# Edit job.yaml to point at your input MP4
transcribe run job.yaml
```

## Output structure

```
outputs/<job_id>/
  job.effective.yaml
  state.json
  segments/
    segment_0000.mp3
    segment_0000.json
    segment_0000.md
    segment_0000_handoff.txt
  final/
    merged.json
    timeline.md
    by_role.md
```

## Legacy mode

The old `--input/--model/--output` usage is still supported and maps to a single-segment run:

```bash
python main.py --input my.mp4 --model gemini-2.0-flash-001 --output transcript.txt
```

## Notes

- The workflow uses the Gemini Developer API Files API for upload/delete per segment.
- If the process is interrupted, rerun `transcribe run job.yaml` to resume.
