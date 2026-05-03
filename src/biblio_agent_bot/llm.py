from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

ProgressCallback = Callable[[str], None]


class LLMProvider(Protocol):
    name: str

    def complete(self, prompt: str, *, schema: type[BaseModel] | None = None) -> str:
        ...


class NullProvider:
    name = "none"

    def complete(self, prompt: str, *, schema: type[BaseModel] | None = None) -> str:
        raise RuntimeError("No LLM provider configured")


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        model: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        model = model or "gemini-3-pro-preview"
        self.model = model
        self.progress = progress or (lambda _message: None)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for provider=gemini")
        from google import genai

        self.client = genai.Client(api_key=api_key)

    def complete(self, prompt: str, *, schema: type[BaseModel] | None = None) -> str:
        config = None
        if schema is not None:
            from google.genai import types

            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=gemini_json_schema(schema),
            )
        self.progress(f"gemini: request started with model {self.model}.")
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        self.progress("gemini: response received.")
        return response.text or ""


class ClaudeCLIProvider:
    name = "claude-cli"

    def __init__(
        self,
        model: str | None = None,
        timeout_seconds: int = 180,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.progress = progress or (lambda _message: None)
        self.binary = _resolve_binary("CLAUDE_BIN", "claude")

    def complete(self, prompt: str, *, schema: type[BaseModel] | None = None) -> str:
        schema_hint = ""
        if schema is not None:
            schema_hint = "\nReturn JSON matching this schema:\n" + json.dumps(
                strict_json_schema(schema), ensure_ascii=False
            )
        cmd = [self.binary, "-p"]
        if self.model:
            cmd.extend(["--model", self.model])
        cmd.extend([
            "--output-format", "json",
            "--permission-mode", "dontAsk",
            "--tools", "",
            prompt + schema_hint,
        ])
        result = _run_with_heartbeat(
            cmd,
            label="claude-cli",
            progress=self.progress,
            timeout_seconds=self.timeout_seconds,
            stdin_text=None,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "claude-cli failed")
        return _unwrap_claude_json(result.stdout)


class CodexCLIProvider:
    name = "codex-cli"

    def __init__(
        self,
        model: str | None = None,
        timeout_seconds: int = 240,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.progress = progress or (lambda _message: None)
        self.binary = _resolve_binary("CODEX_BIN", "codex")

    def complete(self, prompt: str, *, schema: type[BaseModel] | None = None) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            schema_path = tmp / "schema.json"
            output_path = tmp / "last_message.txt"
            cmd = [
                self.binary,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "-C",
                tmpdir,
                "-o",
                str(output_path),
            ]
            if self.model:
                cmd.extend(["--model", self.model])
            if schema is not None:
                schema_path.write_text(json.dumps(strict_json_schema(schema)), encoding="utf-8")
                cmd.extend(["--output-schema", str(schema_path)])
            cmd.append("-")
            result = _run_with_heartbeat(
                cmd,
                label="codex-cli",
                progress=self.progress,
                timeout_seconds=self.timeout_seconds,
                stdin_text=prompt,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "codex-cli failed")
            if output_path.exists():
                return output_path.read_text(encoding="utf-8")
            return result.stdout


def _run_with_heartbeat(
    cmd: list[str],
    *,
    label: str,
    progress: ProgressCallback,
    timeout_seconds: int,
    stdin_text: str | None,
    heartbeat_seconds: int = 15,
) -> subprocess.CompletedProcess[str]:
    start = time.monotonic()
    progress(f"{label}: subprocess started; timeout is {timeout_seconds}s.")
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if stdin_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pending_input = stdin_text
    while True:
        elapsed = time.monotonic() - start
        remaining = max(0.1, timeout_seconds - elapsed)
        try:
            stdout, stderr = process.communicate(
                input=pending_input,
                timeout=min(heartbeat_seconds, remaining),
            )
            duration = int(time.monotonic() - start)
            progress(f"{label}: subprocess finished in {duration}s.")
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        except subprocess.TimeoutExpired:
            pending_input = None
            elapsed = int(time.monotonic() - start)
            if elapsed >= timeout_seconds:
                process.kill()
                stdout, stderr = process.communicate()
                progress(f"{label}: subprocess timed out after {elapsed}s.")
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_seconds, output=stdout, stderr=stderr)
            progress(f"{label}: still running after {elapsed}s...")


def provider_from_name(
    name: str,
    *,
    model_name: str | None = None,
    progress: ProgressCallback | None = None,
) -> LLMProvider:
    if name == "none":
        return NullProvider()
    if name == "gemini":
        return GeminiProvider(model=model_name, progress=progress)
    if name == "claude-cli":
        return ClaudeCLIProvider(model=model_name, progress=progress)
    if name == "codex-cli":
        return CodexCLIProvider(model=model_name, progress=progress)
    raise ValueError(f"Unknown provider: {name}")


def _resolve_binary(env_var: str, executable: str) -> str:
    explicit = os.getenv(env_var)
    if explicit:
        return explicit
    found = shutil.which(executable)
    if found:
        return found
    local = Path.home() / ".local" / "bin" / executable
    if local.exists():
        return str(local)
    raise RuntimeError(f"Could not find {executable}. Set {env_var}.")


def _unwrap_claude_json(stdout: str) -> str:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout
    for key in ["result", "text", "content"]:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return stdout


def strict_json_schema(schema: type[BaseModel]) -> dict:
    raw = schema.model_json_schema()
    _strictify(raw)
    return raw


def gemini_json_schema(schema: type[BaseModel]) -> dict:
    raw = schema.model_json_schema()
    defs = raw.get("$defs", {})
    resolved = _resolve_refs(raw, defs)
    _gemini_sanitize(resolved)
    return resolved


def _strictify(node: object) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties.keys())
        for key in ["properties", "$defs"]:
            value = node.get(key)
            if isinstance(value, dict):
                for child in value.values():
                    _strictify(child)
        for key in ["items", "anyOf", "oneOf", "allOf"]:
            value = node.get(key)
            _strictify(value)
    elif isinstance(node, list):
        for item in node:
            _strictify(item)


def _resolve_refs(node: object, defs: dict) -> object:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.rsplit("/", 1)[-1]
            target = deepcopy(defs[name])
            siblings = {key: value for key, value in node.items() if key != "$ref"}
            if siblings:
                target.update(siblings)
            return _resolve_refs(target, defs)
        return {key: _resolve_refs(value, defs) for key, value in node.items() if key != "$defs"}
    if isinstance(node, list):
        return [_resolve_refs(item, defs) for item in node]
    return node


def _gemini_sanitize(node: object) -> None:
    if isinstance(node, dict):
        node.pop("additionalProperties", None)
        node.pop("$schema", None)
        node.pop("default", None)
        node.pop("examples", None)
        node.pop("title", None)

        any_of = node.get("anyOf")
        if isinstance(any_of, list):
            non_null = [item for item in any_of if not (isinstance(item, dict) and item.get("type") == "null")]
            has_null = len(non_null) != len(any_of)
            if has_null and len(non_null) == 1 and isinstance(non_null[0], dict):
                node.pop("anyOf", None)
                replacement = non_null[0]
                node.update(replacement)
                node["nullable"] = True

        for value in list(node.values()):
            _gemini_sanitize(value)
    elif isinstance(node, list):
        for item in node:
            _gemini_sanitize(item)
