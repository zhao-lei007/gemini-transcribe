from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from google import genai
from google.genai import types


class GeminiClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def upload_file(self, path: str):
        return self.client.files.upload(file=path)

    def delete_file(self, name: str) -> None:
        self.client.files.delete(name=name)

    def generate_json(
        self,
        file_uri: str,
        prompt: str,
        response_schema: Dict[str, Any],
        mime_type: str,
        stream: bool = False,
    ) -> str:
        contents = [types.Part.from_uri(file_uri, mime_type=mime_type), prompt]
        config = types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=response_schema
        )
        if not stream:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            return response.text

        chunks = []
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                chunks.append(chunk.text)
        print()
        return "".join(chunks)

    def repair_json(self, prompt: str, response_schema: Dict[str, Any]) -> str:
        config = types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=response_schema
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=config,
        )
        return response.text


def is_transient_error(error: Exception) -> bool:
    transient_markers = ["timeout", "503", "429", "temporarily", "rate limit"]
    text = str(error).lower()
    return any(marker in text for marker in transient_markers)


def retry_with_backoff(
    func,
    max_retries: int,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
):
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_retries or not is_transient_error(exc):
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            time.sleep(delay)
    raise last_exc


def parse_json(text: str) -> Dict[str, Any]:
    return json.loads(text)
