from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


PROMPT_FILES = {
    "base_instructions": "base_instructions.txt",
    "style_prefs": "style_prefs.txt",
    "role_schema": "role_schema.txt",
    "handoff_schema": "handoff_schema.txt",
}


def load_prompt_bundle(prompts_dir: Path, version: str) -> Dict[str, str]:
    version_dir = prompts_dir / version
    bundle = {}
    for key, filename in PROMPT_FILES.items():
        path = version_dir / filename
        bundle[key] = path.read_text()
    return bundle


def build_prompt(
    bundle: Dict[str, str],
    segment_index: int,
    total_segments: int,
    handoff_summary: Optional[str] = None,
) -> str:
    handoff = handoff_summary or "None"
    return "\n\n".join(
        [
            bundle["base_instructions"],
            bundle["style_prefs"],
            bundle["role_schema"],
            f"Segment context: {segment_index + 1} of {total_segments}.",
            f"Previous handoff summary: {handoff}",
        ]
    )


def build_handoff_prompt(bundle: Dict[str, str], segment_index: int, total_segments: int) -> str:
    return "\n\n".join(
        [
            bundle["handoff_schema"],
            f"You are preparing a handoff for segment {segment_index + 1} of {total_segments}.",
        ]
    )
