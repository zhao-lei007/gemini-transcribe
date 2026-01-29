from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

TIMECODE_RE = re.compile(r"^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$")


def response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "segment_index": {"type": "integer"},
            "transcript": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                        "speaker": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["start", "end", "speaker", "text"],
                },
            },
            "summary": {"type": "string"},
            "roles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["role", "description"],
                },
            },
        },
        "required": ["segment_index", "transcript", "summary", "roles"],
    }


def validate_segment_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    if not isinstance(data, dict):
        return False, ["data is not an object"]
    for key in ["segment_index", "transcript", "summary", "roles"]:
        if key not in data:
            errors.append(f"missing {key}")
    transcript = data.get("transcript", [])
    if not isinstance(transcript, list):
        errors.append("transcript must be list")
    else:
        for idx, item in enumerate(transcript):
            if not isinstance(item, dict):
                errors.append(f"transcript[{idx}] not object")
                continue
            for field in ["start", "end", "speaker", "text"]:
                if field not in item:
                    errors.append(f"transcript[{idx}] missing {field}")
            if "start" in item and not TIMECODE_RE.match(str(item["start"])):
                errors.append(f"transcript[{idx}] invalid start")
            if "end" in item and not TIMECODE_RE.match(str(item["end"])):
                errors.append(f"transcript[{idx}] invalid end")
    roles = data.get("roles", [])
    if not isinstance(roles, list):
        errors.append("roles must be list")
    else:
        for idx, role in enumerate(roles):
            if not isinstance(role, dict):
                errors.append(f"roles[{idx}] not object")
                continue
            if "role" not in role:
                errors.append(f"roles[{idx}] missing role")
            if "description" not in role:
                errors.append(f"roles[{idx}] missing description")
    return len(errors) == 0, errors
