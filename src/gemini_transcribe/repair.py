from __future__ import annotations

import json
from typing import Any, Dict, List


def build_repair_prompt(raw_text: str, errors: List[str], schema: Dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "The following JSON is invalid or does not match the schema.",
            f"Errors: {errors}",
            "Fix the JSON so it matches the schema exactly.",
            f"Schema: {json.dumps(schema)}",
            "Invalid JSON:",
            raw_text,
        ]
    )
