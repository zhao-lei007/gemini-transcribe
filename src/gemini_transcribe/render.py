from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def merge_segments(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"segments": segments}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def write_timeline(path: Path, segments: List[Dict[str, Any]]) -> None:
    lines = []
    for segment in segments:
        for item in segment.get("transcript", []):
            lines.append(
                f"[{item['start']}-{item['end']}] {item['speaker']}: {item['text']}"
            )
    path.write_text("\n".join(lines))


def write_by_role(path: Path, segments: List[Dict[str, Any]]) -> None:
    role_lines: Dict[str, List[str]] = defaultdict(list)
    for segment in segments:
        for item in segment.get("transcript", []):
            role_lines[item["speaker"]].append(
                f"[{item['start']}-{item['end']}] {item['text']}"
            )
    output = []
    for role, lines in role_lines.items():
        output.append(f"## {role}")
        output.extend(lines)
        output.append("")
    path.write_text("\n".join(output).strip())


def write_legacy_text(path: Path, segments: List[Dict[str, Any]]) -> None:
    lines = []
    for segment in segments:
        for item in segment.get("transcript", []):
            lines.append(f"[{item['start']}] {item['speaker']}: {item['text']}")
    path.write_text("\n".join(lines))
