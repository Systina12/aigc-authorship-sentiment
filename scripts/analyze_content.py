from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloader import CommentRecord, DataLoader


DEFAULT_CONFIG_FILE = Path("config/content_analysis_config.json")
DEFAULT_INPUT_FILE = Path("data/cleaned/comments_cleaned.csv")
DEFAULT_OUTPUT_DIR = Path("data/content_analysis")
DEFAULT_OUTPUT_FILE_NAME = "comment_labels.jsonl"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_FILE_NAME

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


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    reasoning_effort: str = ""
    response_format_type: str = "json_object"


@dataclass(frozen=True)
class ContentAnalysisReport:
    loaded_records: int
    written_records: int
    skipped_records: int
    error_records: int
    output_file: Path


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

    successful_records = load_successful_record_hashes(output_path)
    records = DataLoader().load(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written_records = 0
    skipped_records = 0
    error_records = 0
    with output_path.open("a", encoding="utf-8", newline="\n") as output_stream:
        for record_index, record in enumerate(records):
            if limit is not None and record_index >= limit:
                break

            current_record_hash = record_hash(record)
            if current_record_hash in successful_records.get(record_index, set()):
                skipped_records += 1
                continue

            try:
                analysis = analyze_record(record, llm_config)
                output_row = build_success_row(record_index, record, current_record_hash, analysis, llm_config.model)
            except Exception as exc:  # noqa: BLE001 - batch jobs must preserve per-record failures.
                error_records += 1
                output_row = build_error_row(record_index, record, current_record_hash, exc, llm_config.model)

            output_stream.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            output_stream.flush()
            written_records += 1

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

    with config_path.open("r", encoding="utf-8") as config_stream:
        config = json.load(config_stream)
    if not isinstance(config, dict):
        raise ValueError(f"LLM config must be a JSON object: {config_path}")

    llm_config = LLMConfig(
        base_url=str(config.get("base_url", "")),
        api_key=str(config.get("api_key", "")),
        model=str(config.get("model", "")),
        reasoning_effort=str(config.get("reasoning_effort", "")),
        response_format_type=str(config.get("response_format_type", "json_object")),
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


def resolve_output_file(output_file: str | Path | None, output_dir: str | Path | None) -> Path:
    if output_file is not None:
        return Path(output_file)
    if output_dir is not None:
        return Path(output_dir) / DEFAULT_OUTPUT_FILE_NAME
    return DEFAULT_OUTPUT_FILE


def load_successful_record_hashes(output_file: Path) -> dict[int, set[str]]:
    if not output_file.exists():
        return {}

    successful_records: dict[int, set[str]] = {}
    with output_file.open("r", encoding="utf-8") as output_stream:
        for line in output_stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("status") != "ok":
                    continue
                index = int(row["record_index"])
                existing_hash = row.get("record_hash")
                if not isinstance(existing_hash, str):
                    existing_record = row.get("record")
                    if not isinstance(existing_record, dict):
                        continue
                    existing_hash = record_dict_hash(existing_record)
                successful_records.setdefault(index, set()).add(existing_hash)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

    return successful_records


def analyze_record(record: CommentRecord, config: LLMConfig) -> dict[str, Any]:
    payload = build_payload(
        record,
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        response_format_type=config.response_format_type,
    )
    response = requests.post(
        chat_completions_url(config.base_url),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    response_body = response.json()
    content = response_body["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("LLM response content is not a string")

    return normalize_analysis(json.loads(content))


def chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def build_payload(
    record: CommentRecord,
    *,
    model: str,
    reasoning_effort: str = "",
    response_format_type: str = "json_object",
) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"评论内容：{record.content}"},
        ],
    }
    if reasoning_effort.strip():
        payload["reasoning_effort"] = reasoning_effort.strip()

    response_format_type = response_format_type.strip() or "json_object"
    if response_format_type == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "content_analysis",
                "schema": ANALYSIS_SCHEMA,
                "strict": True,
            },
        }
    else:
        payload["response_format"] = {"type": response_format_type}

    return payload


def normalize_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
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
        if not isinstance(rationale, str):
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

    return {
        "summary": summary,
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
