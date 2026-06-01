from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import analyze_sentiment
from scripts.analyze_content import (
    DEFAULT_CONFIG_FILE,
    LLMConfig,
    build_headers,
    extract_response_content,
    llm_endpoint_url,
    load_config,
    normalize_api_type,
)
from scripts import analyze_content


DEFAULT_COMMENT = "AI 绘画效率很高，但训练数据版权问题需要认真讨论。"
DEFAULT_PREVIEW_CHARS = 1200


@dataclass(frozen=True)
class ProviderTestReport:
    request_url: str
    model: str
    task: str
    http_status: int | None
    response_top_level_keys: list[str]
    choice_count: int
    finish_reason: str | None
    message_keys: list[str]
    content_type: str
    content_length: int | None
    content_preview: str
    content_json_ok: bool
    content_json_error: str | None
    normalized_ok: bool
    normalized_error: str | None
    http_error: str | None
    raw_response: dict[str, Any] | None
    raw_text_preview: str

    @property
    def ok(self) -> bool:
        return (
            self.http_status is not None
            and 200 <= self.http_status < 300
            and self.content_json_ok
            and self.normalized_ok
        )


def run_provider_test(
    *,
    config_file: str | Path = DEFAULT_CONFIG_FILE,
    task: str = "content",
    comment: str = DEFAULT_COMMENT,
    base_url: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    response_format_type: str | None = None,
    api_type: str | None = None,
    timeout: int = 60,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
    include_raw_response: bool = False,
) -> ProviderTestReport:
    config = override_config(
        load_config(config_file),
        base_url=base_url,
        model=model,
        reasoning_effort=reasoning_effort,
        response_format_type=response_format_type,
        api_type=api_type,
    )
    payload = build_test_payload(task=task, comment=comment, config=config)
    request_url = llm_endpoint_url(config.base_url, config.api_type)

    response = requests.post(
        request_url,
        headers=build_headers(config.api_key),
        json=payload,
        timeout=timeout,
    )
    http_status = getattr(response, "status_code", None)
    raw_text = getattr(response, "text", "")
    response_body = safe_response_json(response)
    http_error = None
    if http_status is not None and not 200 <= http_status < 300:
        http_error = f"HTTP {http_status}"

    choice = first_choice(response_body)
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = extract_content_for_report(response_body, config.api_type)
    content_text = content if isinstance(content, str) else ""
    content_json, content_json_error = parse_content_json(content_text)
    normalized_error = normalize_response_for_task(task, content_json) if content_json_error is None else "content is not JSON"

    return ProviderTestReport(
        request_url=request_url,
        model=config.model,
        task=task,
        http_status=http_status,
        response_top_level_keys=sorted(response_body.keys()) if isinstance(response_body, dict) else [],
        choice_count=len(response_body.get("choices", [])) if isinstance(response_body, dict) else 0,
        finish_reason=string_or_none(choice.get("finish_reason")),
        message_keys=sorted(message.keys()),
        content_type=type(content).__name__,
        content_length=len(content) if isinstance(content, str) else None,
        content_preview=preview(content_text, preview_chars),
        content_json_ok=content_json_error is None,
        content_json_error=content_json_error,
        normalized_ok=normalized_error is None,
        normalized_error=normalized_error,
        http_error=http_error,
        raw_response=response_body if include_raw_response else None,
        raw_text_preview=preview(raw_text, preview_chars) if isinstance(raw_text, str) else "",
    )


def override_config(
    config: LLMConfig,
    *,
    base_url: str | None,
    model: str | None,
    reasoning_effort: str | None,
    response_format_type: str | None,
    api_type: str | None,
) -> LLMConfig:
    return LLMConfig(
        base_url=base_url if base_url is not None else config.base_url,
        api_key=config.api_key,
        model=model if model is not None else config.model,
        reasoning_effort=reasoning_effort if reasoning_effort is not None else config.reasoning_effort,
        response_format_type=(
            response_format_type if response_format_type is not None else config.response_format_type
        ),
        api_type=api_type if api_type is not None else config.api_type,
        max_workers=config.max_workers,
    )


def build_test_payload(*, task: str, comment: str, config: LLMConfig) -> dict[str, Any]:
    if task == "content":
        payload = analyze_content.build_request_payload(
            analyze_content.CommentRecord("", "", comment, "", 0, "", "", ""),
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            response_format_type=config.response_format_type,
            api_type=config.api_type,
        )
        return maybe_remove_response_format(payload, config.response_format_type)
    if task == "sentiment":
        payload = analyze_sentiment.build_request_payload(
            analyze_sentiment.CommentRecord("", "", comment, "", 0, "", "", ""),
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            response_format_type=config.response_format_type,
            api_type=config.api_type,
        )
        return maybe_remove_response_format(payload, config.response_format_type)
    if task == "minimal":
        payload = build_minimal_payload(config)
        return maybe_remove_response_format(payload, config.response_format_type)
    raise ValueError(f"Unsupported task: {task}")


def build_minimal_payload(config: LLMConfig) -> dict[str, Any]:
    if normalize_api_type(config.api_type) == "responses":
        payload: dict[str, Any] = {
            "model": config.model,
            "temperature": 0,
            "input": [
                {"role": "system", "content": "Return only valid JSON. No Markdown."},
                {"role": "user", "content": 'Return exactly this JSON object: {"ok": true, "note": "provider test"}'},
            ],
        }
        if config.reasoning_effort.strip():
            payload["reasoning"] = {"effort": config.reasoning_effort.strip()}
        response_format_type = config.response_format_type.strip()
        if response_format_type and response_format_type != "none":
            payload["text"] = {"format": {"type": response_format_type}}
        return payload

    payload = {
        "model": config.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Return only valid JSON. No Markdown."},
            {"role": "user", "content": 'Return exactly this JSON object: {"ok": true, "note": "provider test"}'},
        ],
    }
    if config.reasoning_effort.strip():
        payload["reasoning_effort"] = config.reasoning_effort.strip()
    response_format_type = config.response_format_type.strip()
    if response_format_type and response_format_type != "none":
        payload["response_format"] = {"type": response_format_type}
    return payload


def maybe_remove_response_format(payload: dict[str, Any], response_format_type: str) -> dict[str, Any]:
    if response_format_type.strip() == "none":
        payload.pop("response_format", None)
        payload.pop("text", None)
    return payload


def extract_content_for_report(response_body: dict[str, Any], api_type: str) -> str | None:
    try:
        return extract_response_content(response_body, api_type)
    except Exception:
        choice = first_choice(response_body)
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content")
        return content if isinstance(content, str) else None


def safe_response_json(response: requests.Response) -> dict[str, Any]:
    try:
        response_body = response.json()
    except ValueError:
        return {}
    return response_body if isinstance(response_body, dict) else {}


def first_choice(response_body: Any) -> dict[str, Any]:
    if not isinstance(response_body, dict):
        return {}
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    choice = choices[0]
    return choice if isinstance(choice, dict) else {}


def parse_content_json(content: str) -> tuple[Any, str | None]:
    if not content:
        return None, "empty message.content"
    try:
        return json.loads(content), None
    except json.JSONDecodeError as exc:
        candidate = extract_json_candidate(content)
        if candidate != content:
            try:
                return json.loads(candidate), None
            except json.JSONDecodeError:
                pass
        return None, f"{type(exc).__name__}: {exc}"


def extract_json_candidate(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    first_object = stripped.find("{")
    last_object = stripped.rfind("}")
    if 0 <= first_object < last_object:
        return stripped[first_object : last_object + 1]
    return content


def normalize_response_for_task(task: str, value: Any) -> str | None:
    if task == "minimal":
        return None if isinstance(value, dict) else "minimal response is not a JSON object"
    if not isinstance(value, dict):
        return "response content is not a JSON object"
    try:
        if task == "content":
            analyze_content.normalize_analysis(value)
        elif task == "sentiment":
            analyze_sentiment.normalize_sentiment(value)
        else:
            return f"Unsupported task: {task}"
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic tool.
        return f"{type(exc).__name__}: {exc}"
    return None


def string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def preview(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "...<truncated>"


def print_report(report: ProviderTestReport) -> None:
    print(f"Provider URL: {report.request_url}")
    print(f"Model: {report.model}")
    print(f"Task: {report.task}")
    print(f"HTTP status: {report.http_status}")
    if report.http_error:
        print(f"HTTP error: {report.http_error}")
    print(f"Response keys: {', '.join(report.response_top_level_keys) or '(none)'}")
    print(f"Choice count: {report.choice_count}")
    print(f"Finish reason: {report.finish_reason or '(none)'}")
    print(f"Message keys: {', '.join(report.message_keys) or '(none)'}")
    print(f"Content type: {report.content_type}")
    print(f"Content length: {report.content_length}")
    print(f"Content JSON OK: {report.content_json_ok}")
    if report.content_json_error:
        print(f"Content JSON error: {report.content_json_error}")
    print(f"Normalized OK: {report.normalized_ok}")
    if report.normalized_error:
        print(f"Normalized error: {report.normalized_error}")
    print("Content preview:")
    print(report.content_preview or "(empty)")
    if report.raw_text_preview and report.raw_text_preview != report.content_preview:
        print("Raw text preview:")
        print(report.raw_text_preview)
    if report.raw_response is not None:
        print("Raw response:")
        print(json.dumps(report.raw_response, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one diagnostic request to the configured LLM provider.")
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG_FILE))
    parser.add_argument("--task", choices=["content", "sentiment", "minimal"], default="content")
    parser.add_argument("--comment", default=DEFAULT_COMMENT)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Override config reasoning_effort. Use an empty string to omit the field.",
    )
    parser.add_argument(
        "--response-format-type",
        default=None,
        help="Override config response_format_type. Use 'none' to omit response_format for minimal task.",
    )
    parser.add_argument(
        "--api-type",
        choices=["chat_completions", "chat-completions", "responses"],
        default=None,
        help="Override config api_type. Use responses to call /responses instead of /chat/completions.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--preview-chars", type=int, default=DEFAULT_PREVIEW_CHARS)
    parser.add_argument("--show-raw", action="store_true")
    args = parser.parse_args()

    report = run_provider_test(
        config_file=args.config_file,
        task=args.task,
        comment=args.comment,
        base_url=args.base_url,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        response_format_type=args.response_format_type,
        api_type=args.api_type,
        timeout=args.timeout,
        preview_chars=args.preview_chars,
        include_raw_response=args.show_raw,
    )
    print_report(report)
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
