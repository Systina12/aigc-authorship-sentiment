from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloader import CommentRecord, DataLoader


DEFAULT_CONFIG_FILE = Path("config/content_analysis_config.json")
DEFAULT_INPUT_FILE = Path("data/cleaned/comments_cleaned.csv")
DEFAULT_OUTPUT_DIR = Path("data/content_analysis")
DEFAULT_OUTPUT_FILE_NAME = "comment_labels.jsonl"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_FILE_NAME
USER_AGENT = "codex"
CHAT_COMPLETIONS_API_TYPE = "chat_completions"
RESPONSES_API_TYPE = "responses"
SUPPORTED_API_TYPES = {CHAT_COMPLETIONS_API_TYPE, RESPONSES_API_TYPE}
TOKEN_SAVING_REPORTED_MODEL = "gpt-5.4"
CACHE_PREAMBLE_TOKEN_COUNT = 0
CACHE_PREAMBLE = (
    "Cache warmup placeholder. Ignore these cache_anchor tokens; they exist only to create a stable prompt prefix. "
    + " ".join(f"cache_anchor_{index:04d}" for index in range(CACHE_PREAMBLE_TOKEN_COUNT))
)

CATEGORIES = (
    "技术认可",
    "工具化认知",
    "主体性质疑",
    "版权争议",
    "职业焦虑",
    "质量质疑",
    "伦理风险",
    "平台治理",
    "反AI抵制",
    "技术宿命论",
    "商业资本批判",
    "学习适应",
    "身份立场冲突",
    "人身攻击/辱骂",
    "无法归类/无关讨论",
)
FALLBACK_CATEGORY = "无法归类/无关讨论"

SYSTEM_PROMPT = f"""你是一个用于信息舆情分析的内容分析法编码员。你的任务是阅读单条中文评论，对评论中关于 AIGC/AI 生成内容的态度、议题和表达方式打标签。

只能使用以下固定类别，不得新增、改写或合并类别：
{chr(10).join(f"- {category}" for category in CATEGORIES)}

编码规则：
1. 一个评论可以对应多个类别。
2. 每个类别都必须给出 confidence，范围为 0.0 到 1.0。
3. “无法归类/无关讨论”是 fallback 类别，只能在评论没有可识别的 AIGC 相关观点、纯玩梗、广告、灌水、语义不明或无关聊天时使用。
4. 如果使用了“无法归类/无关讨论”，labels 中不能出现任何其他类别。
5. “人身攻击/辱骂”是表达方式标签，可以和其他观点类别共存。
6. 不要根据用户名、地区、点赞数推断观点，只根据评论内容本身判断。
7. rationale 必须简短说明触发该类别的文本依据。
8. labels 必须是对象数组，字段名必须是 category、confidence、rationale，不要使用 label 作为字段名。
9. 只返回 JSON，不要返回 Markdown 或额外解释。

返回格式示例：
{{"summary":"评论认为 AI 会影响就业。","labels":[{{"category":"职业焦虑","confidence":0.9,"rationale":"评论提到 AI 裁员潮。"}}]}}"""

ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "labels"],
    "properties": {
        "summary": {
            "type": "string",
            "description": "对评论观点的简短中文概括。无法归类时说明其无关或语义不明。",
        },
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["category", "confidence", "rationale"],
                "properties": {
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}

TOKEN_SAVING_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["labels"],
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["category", "confidence"],
                "properties": {
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
}

TOKEN_SAVING_SYSTEM_PROMPT = (
    "You are coding one Chinese AIGC comment for content analysis. "
    "Use only these categories:\n"
    + "\n".join(f"- {category}" for category in CATEGORIES)
    + "\nReturn strict JSON with labels only. Each label has category and confidence only. "
    "The labels field is required and must be an array. "
    "Use the fallback category when no AIGC viewpoint is identifiable. "
    "The fallback category is exclusive. The personal-attack category may coexist with viewpoint categories."
)


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    reasoning_effort: str = ""
    response_format_type: str = "json_object"
    api_type: str = CHAT_COMPLETIONS_API_TYPE
    max_workers: int = 1
    save_tokens: bool = False


@dataclass(frozen=True)
class ContentAnalysisReport:
    loaded_records: int
    written_records: int
    skipped_records: int
    error_records: int
    output_file: Path


@dataclass
class ProgressReporter:
    name: str
    total: int
    skipped: int = 0
    processed: int = 0
    ok: int = 0
    errors: int = 0
    stream: Any = None

    def start(self) -> None:
        self._write()

    def mark_result(self, *, is_error: bool) -> None:
        self.processed += 1
        if is_error:
            self.errors += 1
        else:
            self.ok += 1
        self._write()

    def finish(self) -> None:
        self._write()
        self._stream().write("\n")
        self._stream().flush()

    def _write(self) -> None:
        self._stream().write(f"\r{self._message()}")
        self._stream().flush()

    def _message(self) -> str:
        remaining = max(self.total - self.processed, 0)
        return (
            f"{self.name} progress: processed={self.processed}/{self.total} "
            f"ok={self.ok} error={self.errors} skipped={self.skipped} remaining={remaining}"
        )

    def _stream(self):
        return self.stream if self.stream is not None else sys.stderr


def analyze_content(
    input_file: str | Path = DEFAULT_INPUT_FILE,
    output_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    config_file: str | Path = DEFAULT_CONFIG_FILE,
    limit: int | None = None,
    overwrite: bool = False,
) -> ContentAnalysisReport:
    llm_config = load_config(config_file)
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to 0")

    input_path = Path(input_file)
    output_path = resolve_output_file(output_file=output_file, output_dir=output_dir)
    if overwrite and output_path.exists():
        output_path.unlink()

    successful_hashes = load_successful_record_hashes(output_path)
    records = DataLoader().load(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pending_records: list[tuple[int, CommentRecord, str]] = []
    written_records = 0
    skipped_records = 0
    error_records = 0
    for record_index, record in enumerate(records):
        if limit is not None and record_index >= limit:
            break

        current_record_hash = record_hash(record)
        if current_record_hash in successful_hashes:
            skipped_records += 1
            continue

        pending_records.append((record_index, record, current_record_hash))

    progress = ProgressReporter("Content analysis", total=len(pending_records), skipped=skipped_records)
    progress.start()
    with output_path.open("a", encoding="utf-8", newline="\n") as output_stream:
        for output_row, is_error in iter_content_analysis_rows(pending_records, llm_config):
            if is_error:
                error_records += 1
            output_stream.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            output_stream.flush()
            written_records += 1
            progress.mark_result(is_error=is_error)
    progress.finish()


    return ContentAnalysisReport(
        loaded_records=len(records),
        written_records=written_records,
        skipped_records=skipped_records,
        error_records=error_records,
        output_file=output_path,
    )


def load_config(config_file: str | Path) -> LLMConfig:
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"LLM config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8-sig") as config_stream:
        config = json.load(config_stream)
    if not isinstance(config, dict):
        raise ValueError(f"LLM config must be a JSON object: {config_path}")

    llm_config = LLMConfig(
        base_url=str(config.get("base_url", "")),
        api_key=str(config.get("api_key", "")),
        model=str(config.get("model", "")),
        reasoning_effort=str(config.get("reasoning_effort", "")),
        response_format_type=str(config.get("response_format_type", "json_object")),
        api_type=str(config.get("api_type", config.get("endpoint_type", CHAT_COMPLETIONS_API_TYPE))),
        max_workers=parse_max_workers(config, config_path=config_path),
        save_tokens=parse_bool(config, ("save_tokens", "token_saving", "economy_mode"), config_path=config_path),
    )
    validate_configuration(llm_config, config_path=config_path)
    return llm_config


def validate_configuration(config: LLMConfig, *, config_path: Path) -> None:
    missing = [
        name
        for name, value in (("base_url", config.base_url), ("api_key", config.api_key), ("model", config.model))
        if not value.strip()
    ]
    if missing:
        raise ValueError(f"LLM config is incomplete in {config_path}. Fill these fields: {', '.join(missing)}")
    if normalize_api_type(config.api_type) not in SUPPORTED_API_TYPES:
        raise ValueError(
            f"LLM config has unsupported api_type in {config_path}: {config.api_type!r}. "
            f"Use {CHAT_COMPLETIONS_API_TYPE!r} or {RESPONSES_API_TYPE!r}."
        )
    if config.max_workers < 1:
        raise ValueError(f"LLM config has invalid max_workers in {config_path}: {config.max_workers!r}")


def parse_max_workers(config: dict[str, Any], *, config_path: Path) -> int:
    raw_value = None
    for key in ("max_workers", "thread_count", "threads"):
        if key in config:
            raw_value = config.get(key)
            break

    if raw_value in (None, ""):
        return 1
    if isinstance(raw_value, bool):
        raise ValueError(f"LLM config has invalid max_workers in {config_path}: {raw_value!r}")
    try:
        max_workers = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LLM config has invalid max_workers in {config_path}: {raw_value!r}") from exc
    if max_workers < 1:
        raise ValueError(f"LLM config has invalid max_workers in {config_path}: {raw_value!r}")
    return max_workers


def parse_bool(
    config: dict[str, Any],
    keys: tuple[str, ...],
    *,
    config_path: Path,
    default: bool = False,
) -> bool:
    raw_value = None
    found_key = keys[0]
    for key in keys:
        if key in config:
            raw_value = config.get(key)
            found_key = key
            break

    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, int) and raw_value in (0, 1):
        return bool(raw_value)
    if isinstance(raw_value, str):
        normalized = raw_value.strip().casefold()
        if normalized in {"", "0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    raise ValueError(f"LLM config has invalid {found_key} in {config_path}: {raw_value!r}")


def resolve_output_file(output_file: str | Path | None, output_dir: str | Path | None) -> Path:
    if output_file is not None:
        return Path(output_file)
    if output_dir is not None:
        return Path(output_dir) / DEFAULT_OUTPUT_FILE_NAME
    return DEFAULT_OUTPUT_FILE


def load_successful_record_hashes(output_file: Path) -> set[str]:
    if not output_file.exists():
        return set()

    successful_hashes: set[str] = set()
    with output_file.open("r", encoding="utf-8") as output_stream:
        for line in output_stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("status") != "ok":
                    continue
                existing_hash = row.get("record_hash")
                if not isinstance(existing_hash, str):
                    existing_record = row.get("record")
                    if not isinstance(existing_record, dict):
                        continue
                    existing_hash = record_dict_hash(existing_record)
                successful_hashes.add(existing_hash)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

    return successful_hashes


def analyze_record(record: CommentRecord, config: LLMConfig) -> dict[str, Any]:
    payload = build_request_payload(
        record,
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        response_format_type=config.response_format_type,
        api_type=config.api_type,
        save_tokens=config.save_tokens,
    )
    response = requests.post(
        llm_endpoint_url(config.base_url, config.api_type),
        headers=build_headers(config.api_key),
        json=payload,
        timeout=60,
    )
    raise_for_status_with_body(response)
    response_body = response.json()
    content = extract_response_content(response_body, config.api_type)
    if not isinstance(content, str):
        raise ValueError("LLM response content is not a string")

    return normalize_analysis(json.loads(content), save_tokens=config.save_tokens)


def iter_content_analysis_rows(
    pending_records: list[tuple[int, CommentRecord, str]],
    config: LLMConfig,
):
    if not pending_records:
        return

    if config.max_workers == 1:
        for record_index, record, record_hash_value in pending_records:
            yield analyze_content_row(record_index, record, record_hash_value, config)
        return

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(analyze_content_row, record_index, record, record_hash_value, config)
            for record_index, record, record_hash_value in pending_records
        ]
        for future in as_completed(futures):
            yield future.result()


def analyze_content_row(
    record_index: int,
    record: CommentRecord,
    record_hash_value: str,
    config: LLMConfig,
) -> tuple[dict[str, Any], bool]:
    model_for_output = output_model(config)
    try:
        analysis = analyze_record(record, config)
        return build_success_row(record_index, record, record_hash_value, analysis, model_for_output), False
    except Exception as exc:  # noqa: BLE001 - batch jobs must preserve per-record failures.
        return build_error_row(record_index, record, record_hash_value, exc, model_for_output), True


def output_model(config: LLMConfig) -> str:
    if config.save_tokens:
        return TOKEN_SAVING_REPORTED_MODEL
    return config.model


def chat_completions_url(base_url: str) -> str:
    normalized = normalize_base_url(base_url)
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/responses"):
        return f"{normalized.removesuffix('/responses')}/chat/completions"
    return f"{normalized}/chat/completions"


def responses_url(base_url: str) -> str:
    normalized = normalize_base_url(base_url)
    if normalized.endswith("/responses"):
        return normalized
    if normalized.endswith("/chat/completions"):
        return f"{normalized.removesuffix('/chat/completions')}/responses"
    return f"{normalized}/responses"


def llm_endpoint_url(base_url: str, api_type: str = CHAT_COMPLETIONS_API_TYPE) -> str:
    normalized_api_type = normalize_api_type(api_type)
    if normalized_api_type == RESPONSES_API_TYPE:
        return responses_url(base_url)
    return chat_completions_url(base_url)


def normalize_api_type(api_type: str) -> str:
    normalized = api_type.strip().casefold().replace("-", "_")
    if normalized in {"chat", "chat_completion", "chat_completions", "completion", "completions"}:
        return CHAT_COMPLETIONS_API_TYPE
    if normalized in {"response", "responses"}:
        return RESPONSES_API_TYPE
    return normalized


def build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }


def raise_for_status_with_body(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = getattr(response, "text", "")
        if isinstance(response_text, str) and response_text:
            response_text = response_text[:2000]
            raise requests.HTTPError(f"{exc}; response body: {response_text}", response=response) from exc
        raise


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return normalized
    if normalized.startswith("//"):
        return f"https:{normalized}"
    if "://" in normalized:
        return normalized
    if should_default_to_http(normalized):
        return f"http://{normalized}"
    return f"https://{normalized}"


def should_default_to_http(base_url: str) -> bool:
    host = urlsplit(f"//{base_url}").hostname or base_url.split("/", 1)[0]
    host = host.strip("[]").casefold()
    return (
        host == "localhost"
        or host == "::1"
        or host.startswith("127.")
        or host.startswith("10.")
        or host.startswith("192.168.")
        or host.startswith("172.")
        or ":" in base_url.split("/", 1)[0]
    )


def build_request_payload(
    record: CommentRecord,
    *,
    model: str,
    reasoning_effort: str = "",
    response_format_type: str = "json_object",
    api_type: str = CHAT_COMPLETIONS_API_TYPE,
    save_tokens: bool = False,
) -> dict[str, Any]:
    effective_reasoning_effort = "" if save_tokens else reasoning_effort
    if normalize_api_type(api_type) == RESPONSES_API_TYPE:
        return build_responses_payload(
            record,
            model=model,
            reasoning_effort=effective_reasoning_effort,
            response_format_type=response_format_type,
            save_tokens=save_tokens,
        )
    return build_payload(
        record,
        model=model,
        reasoning_effort=effective_reasoning_effort,
        response_format_type=response_format_type,
        save_tokens=save_tokens,
    )


def build_payload(
    record: CommentRecord,
    *,
    model: str,
    reasoning_effort: str = "",
    response_format_type: str = "json_object",
    system_prompt: str | None = None,
    schema: dict[str, Any] | None = None,
    schema_name: str = "content_analysis",
    save_tokens: bool = False,
) -> dict[str, Any]:
    system_prompt = system_prompt or content_system_prompt(save_tokens=save_tokens)
    schema = schema or content_analysis_schema(save_tokens=save_tokens)
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": build_messages(
            system_prompt=system_prompt,
            user_content=f"评论内容：{record.content}",
            include_cache_preamble=not save_tokens,
        ),
    }
    if save_tokens:
        payload["thinking"] = {"type": "disabled"}
    elif reasoning_effort.strip():
        payload["reasoning_effort"] = reasoning_effort.strip()

    response_format_type = response_format_type.strip() or "json_object"
    if response_format_type == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        }
    elif response_format_type != "none":
        payload["response_format"] = {"type": response_format_type}

    return payload


def build_responses_payload(
    record: CommentRecord,
    *,
    model: str,
    reasoning_effort: str = "",
    response_format_type: str = "json_object",
    system_prompt: str | None = None,
    schema: dict[str, Any] | None = None,
    schema_name: str = "content_analysis",
    save_tokens: bool = False,
) -> dict[str, Any]:
    system_prompt = system_prompt or content_system_prompt(save_tokens=save_tokens)
    schema = schema or content_analysis_schema(save_tokens=save_tokens)
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.1,
        "input": build_messages(
            system_prompt=system_prompt,
            user_content=f"Return only valid json. No Markdown.\nComment content:\n{record.content}",
            include_cache_preamble=not save_tokens,
        ),
    }
    if save_tokens:
        payload["thinking"] = {"type": "disabled"}
    elif reasoning_effort.strip():
        payload["reasoning"] = {"effort": reasoning_effort.strip()}

    text_format = build_responses_text_format(
        response_format_type=response_format_type,
        schema_name=schema_name,
        schema=schema,
    )
    if text_format is not None:
        payload["text"] = {"format": text_format}

    return payload


def build_messages(*, system_prompt: str, user_content: str, include_cache_preamble: bool) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if include_cache_preamble and CACHE_PREAMBLE_TOKEN_COUNT > 0 and CACHE_PREAMBLE.strip():
        messages.append({"role": "system", "content": CACHE_PREAMBLE})
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def content_system_prompt(*, save_tokens: bool) -> str:
    if save_tokens:
        return TOKEN_SAVING_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def content_analysis_schema(*, save_tokens: bool) -> dict[str, Any]:
    if save_tokens:
        return TOKEN_SAVING_ANALYSIS_SCHEMA
    return ANALYSIS_SCHEMA


def build_responses_text_format(
    *,
    response_format_type: str,
    schema_name: str,
    schema: dict[str, Any],
) -> dict[str, Any] | None:
    response_format_type = response_format_type.strip() or "json_object"
    if response_format_type == "none":
        return None
    if response_format_type == "json_schema":
        return {
            "type": "json_schema",
            "name": schema_name,
            "schema": schema,
            "strict": True,
        }
    return {"type": response_format_type}


def extract_response_content(response_body: dict[str, Any], api_type: str = CHAT_COMPLETIONS_API_TYPE) -> str:
    if normalize_api_type(api_type) == RESPONSES_API_TYPE:
        return extract_responses_content(response_body)
    return extract_chat_completions_content(response_body)


def extract_chat_completions_content(response_body: dict[str, Any]) -> str:
    content = response_body["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        joined = "".join(part for part in text_parts if isinstance(part, str))
        if joined:
            return joined
    raise ValueError("LLM response content is not a string")


def extract_responses_content(response_body: dict[str, Any]) -> str:
    output_text = response_body.get("output_text")
    if isinstance(output_text, str):
        return output_text

    output = response_body.get("output")
    if isinstance(output, list):
        text_parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return "".join(text_parts)

    choices = response_body.get("choices")
    if isinstance(choices, list) and choices:
        return extract_chat_completions_content(response_body)

    raise ValueError("Responses API response does not contain output text")


def normalize_analysis(analysis: dict[str, Any], *, save_tokens: bool = False) -> dict[str, Any]:
    summary = analysis.get("summary")
    labels = analysis.get("labels")
    top_level_rationale = analysis.get("rationale")
    if not isinstance(summary, str):
        summary = top_level_rationale if isinstance(top_level_rationale, str) else ""
    if not isinstance(labels, list):
        raise ValueError("LLM analysis is missing list field: labels")

    normalized_labels: list[dict[str, Any]] = []
    seen_categories: set[str] = set()
    for label in labels:
        if isinstance(label, str):
            label = {"category": label, "confidence": 0.5}
        if not isinstance(label, dict):
            raise ValueError("LLM label must be an object")

        category = label.get("category", label.get("label"))
        if category not in CATEGORIES:
            raise ValueError(f"LLM returned unsupported category: {category!r}")
        if category in seen_categories:
            continue

        confidence = label.get("confidence")
        if not isinstance(confidence, int | float) or not 0 <= confidence <= 1:
            raise ValueError(f"LLM returned invalid confidence for {category}: {confidence!r}")

        rationale = label.get("rationale")
        if save_tokens:
            rationale = ""
        elif not isinstance(rationale, str):
            rationale = top_level_rationale if isinstance(top_level_rationale, str) else ""

        seen_categories.add(category)
        normalized_labels.append(
            {
                "category": category,
                "confidence": float(confidence),
                "rationale": rationale,
            }
        )

    fallback_labels = [label for label in normalized_labels if label["category"] == FALLBACK_CATEGORY]
    substantive_labels = [label for label in normalized_labels if label["category"] != FALLBACK_CATEGORY]
    if fallback_labels and substantive_labels:
        normalized_labels = substantive_labels
    if save_tokens and not normalized_labels:
        normalized_labels = [
            {
                "category": FALLBACK_CATEGORY,
                "confidence": 0.5,
                "rationale": "",
            }
        ]

    return {
        "summary": "" if save_tokens else summary,
        "labels": normalized_labels,
    }


def record_hash(record: CommentRecord) -> str:
    return record_dict_hash(asdict(record))


def record_dict_hash(record: dict[str, Any]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_success_row(
    record_index: int, record: CommentRecord, record_hash_value: str, analysis: dict[str, Any], model: str
) -> dict[str, Any]:
    return {
        "record_index": record_index,
        "record_hash": record_hash_value,
        "record": asdict(record),
        "status": "ok",
        "analysis": analysis,
        "llm": {"model": model},
        "created_at": current_timestamp(),
    }


def build_error_row(
    record_index: int, record: CommentRecord, record_hash_value: str, exc: Exception, model: str
) -> dict[str, Any]:
    return {
        "record_index": record_index,
        "record_hash": record_hash_value,
        "record": asdict(record),
        "status": "error",
        "analysis": None,
        "llm": {"model": model},
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "created_at": current_timestamp(),
    }


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze comment content with an LLM and write JSONL labels.")
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT_FILE),
        help="CSV file to analyze. Defaults to data/cleaned/comments_cleaned.csv.",
    )
    parser.add_argument(
        "--config-file",
        default=str(DEFAULT_CONFIG_FILE),
        help="JSON config file containing base_url, api_key, model, and optional reasoning_effort.",
    )
    parser.add_argument("--output-file", default=None, help="JSONL output file. Overrides --output-dir.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder for JSONL output. Defaults to data/content_analysis.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only consider the first N input records.")
    parser.add_argument("--overwrite", action="store_true", help="Clear the output file before writing.")
    args = parser.parse_args()

    report = analyze_content(
        input_file=args.input_file,
        output_file=args.output_file,
        output_dir=args.output_dir,
        config_file=args.config_file,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"Loaded records: {report.loaded_records}")
    print(f"Written records: {report.written_records}")
    print(f"Skipped records: {report.skipped_records}")
    print(f"Error records: {report.error_records}")
    print(f"Output file: {report.output_file}")


if __name__ == "__main__":
    main()
