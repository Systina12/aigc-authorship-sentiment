from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloader import CommentRecord, DataLoader
from scripts.analyze_content import (
    DEFAULT_CONFIG_FILE,
    LLMConfig,
    ProgressReporter,
    build_headers,
    build_payload as build_content_payload,
    build_responses_payload as build_content_responses_payload,
    current_timestamp,
    extract_response_content,
    load_config,
    load_successful_record_hashes,
    llm_endpoint_url,
    normalize_api_type,
    output_model,
    raise_for_status_with_body,
    record_hash,
    RESPONSES_API_TYPE,
)


DEFAULT_INPUT_FILE = Path("data/cleaned/comments_cleaned.csv")
DEFAULT_OUTPUT_DIR = Path("data/sentiment_analysis")
DEFAULT_OUTPUT_FILE_NAME = "comment_sentiment.jsonl"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_FILE_NAME

SENTIMENT_CATEGORIES = (
    "乐观",
    "焦虑",
    "防御",
    "愤怒",
    "嘲讽",
    "质疑",
    "中性/无法判断",
)
FALLBACK_CATEGORY = "中性/无法判断"
SENTIMENT_POLARITIES = ("positive", "negative", "mixed", "neutral")

SYSTEM_PROMPT = f"""你是一个用于中文评论研究的情绪与语气分析员。你的任务是阅读单条评论，判断评论中的情绪、语气和互动立场。

只能使用以下固定类别，不得新增、改写或合并类别：
{chr(10).join(f"- {category}" for category in SENTIMENT_CATEGORIES)}

类别说明：
- 乐观：对 AI/AIGC、未来发展、工具使用、行业变化表达期待、认可或积极态度。
- 焦虑：表达担忧、恐惧、不安、失业压力、前景不确定或被替代感。
- 防御：替 AI、创作者、平台、某类用户或某种立场辩护，反驳批评，或为相关行为正当化。
- 愤怒：明显表达愤怒、攻击性不满、强烈谴责或敌意。
- 嘲讽：使用反讽、挖苦、阴阳怪气、调侃、贬损式比喻等语气。
- 质疑：提出怀疑、否定、追问、不信任或认为说法/作品/技术站不住脚。
- 中性/无法判断：没有明显情绪、语气或态度，语义不明，或与情绪分析无关。

编码规则：
1. 一个评论可以对应多个类别。
2. 每个类别都必须给出 confidence，范围为 0.0 到 1.0。
3. “中性/无法判断”是 fallback 类别，只能在没有明显情绪、语气或态度时使用。
4. 如果使用了“中性/无法判断”，labels 中不能出现任何其他类别。
5. dominant_category 必须从上述类别中选择，表示最主导的情绪或语气。
6. sentiment_polarity 必须是 positive、negative、mixed、neutral 之一。
7. labels 必须是对象数组，字段名必须是 category、confidence、rationale，不要使用 label 作为字段名。
8. 只根据评论内容本身判断，不要根据用户名、地区、点赞数推断。
9. 只返回 JSON，不要返回 Markdown 或额外解释。

返回格式示例：
{{"summary":"评论担心 AI 造成失业。","dominant_category":"焦虑","sentiment_polarity":"negative","labels":[{{"category":"焦虑","confidence":0.9,"rationale":"评论提到失业压力。"}}]}}"""

SENTIMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "dominant_category", "sentiment_polarity", "labels"],
    "properties": {
        "summary": {"type": "string"},
        "dominant_category": {"type": "string", "enum": list(SENTIMENT_CATEGORIES)},
        "sentiment_polarity": {"type": "string", "enum": list(SENTIMENT_POLARITIES)},
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["category", "confidence", "rationale"],
                "properties": {
                    "category": {"type": "string", "enum": list(SENTIMENT_CATEGORIES)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}

TOKEN_SAVING_SENTIMENT_SCHEMA: dict[str, Any] = {
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
                    "category": {"type": "string", "enum": list(SENTIMENT_CATEGORIES)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
}

TOKEN_SAVING_SYSTEM_PROMPT = (
    "You are coding one Chinese comment for emotion and tone analysis. "
    "Use only these categories:\n"
    + "\n".join(f"- {category}" for category in SENTIMENT_CATEGORIES)
    + "\nReturn strict JSON with labels only. "
    "The labels field is required and must be an array. "
    "Each label has category and confidence only. Use the fallback category when tone is unclear. "
    "The fallback category is exclusive."
)


@dataclass(frozen=True)
class SentimentAnalysisReport:
    loaded_records: int
    written_records: int
    skipped_records: int
    error_records: int
    output_file: Path


def analyze_sentiment(
    input_file: str | Path = DEFAULT_INPUT_FILE,
    output_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    config_file: str | Path = DEFAULT_CONFIG_FILE,
    limit: int | None = None,
    overwrite: bool = False,
) -> SentimentAnalysisReport:
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

    progress = ProgressReporter("Sentiment analysis", total=len(pending_records), skipped=skipped_records)
    progress.start()
    with output_path.open("a", encoding="utf-8", newline="\n") as output_stream:
        for output_row, is_error in iter_sentiment_analysis_rows(pending_records, llm_config):
            if is_error:
                error_records += 1
            output_stream.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            output_stream.flush()
            written_records += 1
            progress.mark_result(is_error=is_error)
    progress.finish()

    return SentimentAnalysisReport(
        loaded_records=len(records),
        written_records=written_records,
        skipped_records=skipped_records,
        error_records=error_records,
        output_file=output_path,
    )


def resolve_output_file(output_file: str | Path | None, output_dir: str | Path | None) -> Path:
    if output_file is not None:
        return Path(output_file)
    if output_dir is not None:
        return Path(output_dir) / DEFAULT_OUTPUT_FILE_NAME
    return DEFAULT_OUTPUT_FILE


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

    return normalize_sentiment(json.loads(content), save_tokens=config.save_tokens)


def iter_sentiment_analysis_rows(
    pending_records: list[tuple[int, CommentRecord, str]],
    config: LLMConfig,
):
    if not pending_records:
        return

    if config.max_workers == 1:
        for record_index, record, record_hash_value in pending_records:
            yield analyze_sentiment_row(record_index, record, record_hash_value, config)
        return

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(analyze_sentiment_row, record_index, record, record_hash_value, config)
            for record_index, record, record_hash_value in pending_records
        ]
        for future in as_completed(futures):
            yield future.result()


def analyze_sentiment_row(
    record_index: int,
    record: CommentRecord,
    record_hash_value: str,
    config: LLMConfig,
) -> tuple[dict[str, Any], bool]:
    model_for_output = output_model(config)
    try:
        sentiment = analyze_record(record, config)
        return build_success_row(record_index, record, record_hash_value, sentiment, model_for_output), False
    except Exception as exc:  # noqa: BLE001 - batch jobs must preserve per-record failures.
        return build_error_row(record_index, record, record_hash_value, exc, model_for_output), True


def build_request_payload(
    record: CommentRecord,
    *,
    model: str,
    reasoning_effort: str = "",
    response_format_type: str = "json_object",
    api_type: str = "",
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
    save_tokens: bool = False,
) -> dict[str, Any]:
    return build_content_payload(
        record,
        model=model,
        reasoning_effort=reasoning_effort,
        response_format_type=response_format_type,
        system_prompt=sentiment_system_prompt(save_tokens=save_tokens),
        schema=sentiment_schema(save_tokens=save_tokens),
        schema_name="sentiment_analysis",
        save_tokens=save_tokens,
    )


def build_responses_payload(
    record: CommentRecord,
    *,
    model: str,
    reasoning_effort: str = "",
    response_format_type: str = "json_object",
    save_tokens: bool = False,
) -> dict[str, Any]:
    return build_content_responses_payload(
        record,
        model=model,
        reasoning_effort=reasoning_effort,
        response_format_type=response_format_type,
        system_prompt=sentiment_system_prompt(save_tokens=save_tokens),
        schema=sentiment_schema(save_tokens=save_tokens),
        schema_name="sentiment_analysis",
        save_tokens=save_tokens,
    )



def sentiment_system_prompt(*, save_tokens: bool) -> str:
    if save_tokens:
        return TOKEN_SAVING_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def sentiment_schema(*, save_tokens: bool) -> dict[str, Any]:
    if save_tokens:
        return TOKEN_SAVING_SENTIMENT_SCHEMA
    return SENTIMENT_SCHEMA


def normalize_sentiment(sentiment: dict[str, Any], *, save_tokens: bool = False) -> dict[str, Any]:
    top_level_rationale = sentiment.get("rationale")
    summary = sentiment.get("summary")
    labels = sentiment.get("labels")
    if not isinstance(summary, str):
        summary = top_level_rationale if isinstance(top_level_rationale, str) else ""
    if not isinstance(labels, list):
        raise ValueError("LLM sentiment is missing list field: labels")

    normalized_labels = normalize_labels(labels, top_level_rationale=top_level_rationale, save_tokens=save_tokens)
    if not normalized_labels:
        normalized_labels = [
            {
                "category": FALLBACK_CATEGORY,
                "confidence": 0.5,
                "rationale": "" if save_tokens else "模型未返回明确情绪或语气标签。",
            }
        ]

    fallback_labels = [label for label in normalized_labels if label["category"] == FALLBACK_CATEGORY]
    substantive_labels = [label for label in normalized_labels if label["category"] != FALLBACK_CATEGORY]
    if fallback_labels and substantive_labels:
        normalized_labels = substantive_labels

    dominant_category = sentiment.get("dominant_category", sentiment.get("dominant_label"))
    if dominant_category not in SENTIMENT_CATEGORIES or (
        dominant_category == FALLBACK_CATEGORY and any(label["category"] != FALLBACK_CATEGORY for label in normalized_labels)
    ):
        dominant_category = max(normalized_labels, key=lambda label: label["confidence"])["category"]

    sentiment_polarity = sentiment.get("sentiment_polarity", sentiment.get("polarity"))
    if sentiment_polarity not in SENTIMENT_POLARITIES:
        sentiment_polarity = infer_polarity(dominant_category, normalized_labels)

    if dominant_category == FALLBACK_CATEGORY:
        sentiment_polarity = "neutral"

    return {
        "summary": "" if save_tokens else summary,
        "dominant_category": dominant_category,
        "sentiment_polarity": sentiment_polarity,
        "labels": normalized_labels,
    }


def normalize_labels(
    labels: list[Any], *, top_level_rationale: object, save_tokens: bool = False
) -> list[dict[str, Any]]:
    normalized_labels: list[dict[str, Any]] = []
    seen_categories: set[str] = set()
    for label in labels:
        if isinstance(label, str):
            label = {"category": label, "confidence": 0.5}
        if not isinstance(label, dict):
            raise ValueError("LLM label must be an object")

        category = label.get("category", label.get("label"))
        if category not in SENTIMENT_CATEGORIES:
            raise ValueError(f"LLM returned unsupported sentiment category: {category!r}")
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

    return normalized_labels


def infer_polarity(dominant_category: str, labels: list[dict[str, Any]]) -> str:
    categories = {label["category"] for label in labels}
    if "乐观" in categories and categories - {"乐观", FALLBACK_CATEGORY}:
        return "mixed"
    if dominant_category == "乐观":
        return "positive"
    if dominant_category == FALLBACK_CATEGORY:
        return "neutral"
    return "negative"


def build_success_row(
    record_index: int, record: CommentRecord, record_hash_value: str, sentiment: dict[str, Any], model: str
) -> dict[str, Any]:
    return {
        "record_index": record_index,
        "record_hash": record_hash_value,
        "record": asdict(record),
        "status": "ok",
        "sentiment": sentiment,
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
        "sentiment": None,
        "llm": {"model": model},
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "created_at": current_timestamp(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze comment emotion and tone with an LLM.")
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
        help="Folder for JSONL output. Defaults to data/sentiment_analysis.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only consider the first N input records.")
    parser.add_argument("--overwrite", action="store_true", help="Clear the output file before writing.")
    args = parser.parse_args()

    report = analyze_sentiment(
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
