from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def parse_model(text: str, model: type[T]) -> T:
    data = parse_json(text)
    return model.model_validate(data)


def parse_json(text: str):
    text = text.strip()
    if not text:
        raise ValueError("Empty JSON response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    first = min([i for i in [text.find("{"), text.find("[")] if i >= 0], default=-1)
    if first < 0:
        raise ValueError("No JSON object or list found")
    candidate = text[first:]
    return json.loads(candidate)
