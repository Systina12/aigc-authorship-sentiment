from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import jieba
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib import font_manager
from pyecharts import options as opts
from pyecharts.charts import Bar, Graph as PyeGraph, HeatMap, Page, Pie
from wordcloud import WordCloud

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloader import DataLoader
from scripts.analyze_content import record_dict_hash, record_hash


DEFAULT_CLEANED_FILE = Path("data/cleaned/comments_cleaned.csv")
DEFAULT_CONTENT_FILE = Path("data/content_analysis/comment_labels.jsonl")
DEFAULT_SENTIMENT_FILE = Path("data/sentiment_analysis/comment_sentiment.jsonl")
DEFAULT_TOPIC_FILE = Path("data/topic_clustering/comment_topics.csv")
DEFAULT_TOPIC_INFO_FILE = Path("data/topic_clustering/topic_info.csv")
DEFAULT_COOCCURRENCE_DIR = Path("data/cooccurrence_analysis")
DEFAULT_OUTPUT_DIR = Path("data/analysis_report")

TABLE_FILE_NAMES = (
    "content_label_summary.csv",
    "sentiment_label_summary.csv",
    "sentiment_polarity_summary.csv",
    "dominant_sentiment_summary.csv",
    "content_sentiment_crosstab.csv",
    "content_polarity_crosstab.csv",
    "topic_summary.csv",
    "topic_content_crosstab.csv",
    "topic_sentiment_crosstab.csv",
    "word_frequency.csv",
    "data_quality_summary.csv",
)
FIGURE_FILE_NAMES = (
    "content_label_distribution.png",
    "sentiment_label_distribution.png",
    "sentiment_polarity_distribution.png",
    "content_sentiment_heatmap.png",
    "topic_distribution.png",
    "wordcloud.png",
    "content_sentiment_network.png",
)
HTML_FILE_NAMES = ("report.html", "interactive_report.html")
METADATA_FILE_NAME = "report_metadata.json"

CONTENT_FALLBACK_CATEGORY = "无法归类/无关讨论"
SENTIMENT_FALLBACK_CATEGORY = "中性/无法判断"
NOISE_TOPIC = "-1"

BUILTIN_STOPWORDS = {
    "一个",
    "一些",
    "不是",
    "不能",
    "不要",
    "什么",
    "他们",
    "你们",
    "我们",
    "这个",
    "这些",
    "那个",
    "那些",
    "可以",
    "就是",
    "还是",
    "没有",
    "然后",
    "因为",
    "所以",
    "但是",
    "如果",
    "已经",
    "自己",
    "真的",
    "怎么",
    "这么",
    "那么",
    "觉得",
    "应该",
    "可能",
    "以及",
    "进行",
    "对于",
    "不是",
    "哈哈",
    "哈哈哈",
    "ai",
    "AI",
}
TOKEN_PUNCTUATION_RE = re.compile(r"^[\W_]+$", re.UNICODE)


@dataclass(frozen=True)
class Label:
    category: str
    confidence: float


@dataclass(frozen=True)
class AnalysisRecord:
    record_index: int | None
    record_hash: str | None
    labels: tuple[Label, ...] = ()
    dominant_category: str | None = None
    sentiment_polarity: str | None = None


@dataclass(frozen=True)
class TopicRecord:
    record_index: int
    topic: str
    topic_label: str
    probability: float | None


@dataclass(frozen=True)
class JsonlLoadResult:
    input_rows: int = 0
    ok_rows: int = 0
    error_rows: int = 0
    stale_rows: int = 0
    duplicate_ok_rows: int = 0
    records: list[AnalysisRecord] = field(default_factory=list)


@dataclass(frozen=True)
class TopicLoadResult:
    input_rows: int = 0
    topic_rows: int = 0
    noise_rows: int = 0
    records: list[TopicRecord] = field(default_factory=list)


@dataclass(frozen=True)
class ReportGenerationReport:
    output_dir: Path
    tables: dict[str, Path]
    figures: dict[str, Path]
    static_html: Path
    interactive_html: Path
    metadata_file: Path


def build_analysis_report(
    *,
    cleaned_file: str | Path = DEFAULT_CLEANED_FILE,
    content_file: str | Path = DEFAULT_CONTENT_FILE,
    sentiment_file: str | Path = DEFAULT_SENTIMENT_FILE,
    topic_file: str | Path = DEFAULT_TOPIC_FILE,
    topic_info_file: str | Path = DEFAULT_TOPIC_INFO_FILE,
    cooccurrence_dir: str | Path = DEFAULT_COOCCURRENCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    top_n: int = 20,
    min_confidence: float = 0.0,
    min_sentiment_confidence: float | None = None,
    stopwords_file: str | Path | None = None,
) -> ReportGenerationReport:
    validate_parameters(limit=limit, top_n=top_n, min_confidence=min_confidence, min_sentiment_confidence=min_sentiment_confidence)

    cleaned_path = Path(cleaned_file)
    content_path = Path(content_file)
    sentiment_path = Path(sentiment_file)
    topic_path = Path(topic_file)
    topic_info_path = Path(topic_info_file)
    cooccurrence_path = Path(cooccurrence_dir)
    output_path = Path(output_dir)
    figures_path = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    cleanup_managed_outputs(output_path)

    warnings: list[str] = []
    input_status: dict[str, dict[str, Any]] = {}
    sentiment_threshold = min_confidence if min_sentiment_confidence is None else min_sentiment_confidence

    font_path, font_name = configure_chinese_font(warnings)

    cleaned_records = load_cleaned_records(cleaned_path, limit=limit, input_status=input_status, warnings=warnings)
    allowed_hashes, allowed_indices, current_index_by_hash = build_current_record_scope(
        cleaned_records, input_status["cleaned_file"]
    )
    content_data = load_analysis_file(
        content_path,
        section_name="analysis",
        min_confidence=min_confidence,
        limit=limit,
        allowed_hashes=allowed_hashes,
        allowed_indices=allowed_indices,
        current_index_by_hash=current_index_by_hash,
        input_status=input_status,
        input_name="content_file",
        warnings=warnings,
    )
    sentiment_data = load_analysis_file(
        sentiment_path,
        section_name="sentiment",
        min_confidence=sentiment_threshold,
        limit=limit,
        allowed_hashes=allowed_hashes,
        allowed_indices=allowed_indices,
        current_index_by_hash=current_index_by_hash,
        input_status=input_status,
        input_name="sentiment_file",
        warnings=warnings,
    )
    topic_data = load_topic_file(
        topic_path,
        topic_info_path,
        limit=limit,
        input_status=input_status,
        warnings=warnings,
    )
    input_status["cooccurrence_dir"] = describe_path(cooccurrence_path, expected_kind="dir")

    content_records = content_data.records
    sentiment_records = sentiment_data.records
    topic_records = topic_data.records
    matched_content_sentiment = match_analysis_records(content_records, sentiment_records)

    tables: dict[str, pd.DataFrame] = {}
    tables["content_label_summary.csv"] = summarize_labels(content_records, denominator=len(content_records))
    tables["sentiment_label_summary.csv"] = summarize_labels(sentiment_records, denominator=len(sentiment_records))
    tables["sentiment_polarity_summary.csv"] = summarize_single_field(
        [record.sentiment_polarity for record in sentiment_records], field_name="polarity", denominator=len(sentiment_records)
    )
    tables["dominant_sentiment_summary.csv"] = summarize_single_field(
        [record.dominant_category for record in sentiment_records],
        field_name="category",
        denominator=len(sentiment_records),
    )
    tables["content_sentiment_crosstab.csv"] = build_analysis_crosstab(
        matched_content_sentiment,
        left_name="content_label",
        right_values=lambda pair: [label.category for label in pair[1].labels],
    )
    tables["content_polarity_crosstab.csv"] = build_analysis_crosstab(
        matched_content_sentiment,
        left_name="content_label",
        right_values=lambda pair: [pair[1].sentiment_polarity] if pair[1].sentiment_polarity else [],
    )

    if input_status["topic_file"]["status"] == "loaded":
        topic_by_index = {record.record_index: record for record in topic_records}
        content_by_index = records_by_index(content_records)
        sentiment_by_index = records_by_index(sentiment_records)
        matched_topic_content = [
            (topic_by_index[index], content_by_index[index])
            for index in sorted(topic_by_index.keys() & content_by_index.keys())
        ]
        matched_topic_sentiment = [
            (topic_by_index[index], sentiment_by_index[index])
            for index in sorted(topic_by_index.keys() & sentiment_by_index.keys())
        ]
        tables["topic_summary.csv"] = summarize_topics(topic_records)
        tables["topic_content_crosstab.csv"] = build_topic_crosstab(matched_topic_content)
        tables["topic_sentiment_crosstab.csv"] = build_topic_crosstab(matched_topic_sentiment)
    else:
        matched_topic_content = []
        matched_topic_sentiment = []

    word_frequency = build_word_frequency(cleaned_records, load_stopwords(stopwords_file), warnings=warnings)
    if input_status["cleaned_file"]["status"] == "loaded":
        tables["word_frequency.csv"] = word_frequency

    quality_metrics = build_quality_metrics(
        cleaned_records=cleaned_records,
        content_data=content_data,
        sentiment_data=sentiment_data,
        topic_data=topic_data,
        content_records=content_records,
        sentiment_records=sentiment_records,
        matched_content_sentiment=matched_content_sentiment,
        matched_topic_content=matched_topic_content,
        matched_topic_sentiment=matched_topic_sentiment,
    )
    tables["data_quality_summary.csv"] = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in quality_metrics.items()]
    )

    table_paths = write_tables(tables, output_path)
    figure_paths = write_figures(
        tables=tables,
        cooccurrence_dir=cooccurrence_path,
        figures_dir=figures_path,
        top_n=top_n,
        font_path=font_path,
        warnings=warnings,
    )

    static_html = output_path / "report.html"
    interactive_html = output_path / "interactive_report.html"
    metadata_file = output_path / "report_metadata.json"
    write_static_html(static_html, tables=tables, figures=figure_paths, input_status=input_status, quality_metrics=quality_metrics)
    write_interactive_html(interactive_html, tables=tables, cooccurrence_dir=cooccurrence_path, top_n=top_n, warnings=warnings)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "limit": limit,
            "top_n": top_n,
            "min_confidence": min_confidence,
            "min_sentiment_confidence": sentiment_threshold,
            "stopwords_file": str(stopwords_file) if stopwords_file is not None else None,
        },
        "font": {"name": font_name, "path": str(font_path) if font_path else None},
        "inputs": input_status,
        "outputs": {
            "tables": {name: str(path) for name, path in table_paths.items()},
            "figures": {name: str(path) for name, path in figure_paths.items()},
            "report_html": str(static_html),
            "interactive_report_html": str(interactive_html),
        },
        "quality_metrics": quality_metrics,
        "warnings": warnings,
    }
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return ReportGenerationReport(
        output_dir=output_path,
        tables=table_paths,
        figures=figure_paths,
        static_html=static_html,
        interactive_html=interactive_html,
        metadata_file=metadata_file,
    )


def validate_parameters(*, limit: int | None, top_n: int, min_confidence: float, min_sentiment_confidence: float | None) -> None:
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to 0")
    if top_n < 1:
        raise ValueError("top_n must be greater than or equal to 1")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0.0 and 1.0")
    if min_sentiment_confidence is not None and not 0 <= min_sentiment_confidence <= 1:
        raise ValueError("min_sentiment_confidence must be between 0.0 and 1.0")


def cleanup_managed_outputs(output_dir: Path) -> None:
    for file_name in TABLE_FILE_NAMES:
        delete_file_if_exists(output_dir / file_name)
    for file_name in HTML_FILE_NAMES:
        delete_file_if_exists(output_dir / file_name)
    delete_file_if_exists(output_dir / METADATA_FILE_NAME)

    figures_dir = output_dir / "figures"
    for file_name in FIGURE_FILE_NAMES:
        delete_file_if_exists(figures_dir / file_name)


def delete_file_if_exists(path: Path) -> None:
    if path.is_file():
        path.unlink()


def load_cleaned_records(
    cleaned_file: Path,
    *,
    limit: int | None,
    input_status: dict[str, dict[str, Any]],
    warnings: list[str],
) -> list[Any]:
    input_status["cleaned_file"] = describe_path(cleaned_file, expected_kind="file")
    if not cleaned_file.exists():
        return []

    try:
        records = DataLoader().load(cleaned_file)
        if limit is not None:
            records = records[:limit]
        input_status["cleaned_file"].update({"status": "loaded", "records": len(records)})
        return records
    except Exception as exc:  # noqa: BLE001 - report generation should explain bad inputs.
        input_status["cleaned_file"].update({"status": "error", "error": str(exc)})
        warnings.append(f"Failed to load cleaned data: {exc}")
        return []


def build_current_record_scope(
    cleaned_records: list[Any],
    cleaned_status: dict[str, Any],
) -> tuple[set[str] | None, set[int] | None, dict[str, int] | None]:
    if cleaned_status.get("status") != "loaded":
        return None, None, None

    hash_by_index = [record_hash(record) for record in cleaned_records]
    return set(hash_by_index), set(range(len(cleaned_records))), {value: index for index, value in enumerate(hash_by_index)}


def load_analysis_file(
    input_file: Path,
    *,
    section_name: str,
    min_confidence: float,
    limit: int | None,
    allowed_hashes: set[str] | None,
    allowed_indices: set[int] | None,
    current_index_by_hash: dict[str, int] | None,
    input_status: dict[str, dict[str, Any]],
    input_name: str,
    warnings: list[str],
) -> JsonlLoadResult:
    input_status[input_name] = describe_path(input_file, expected_kind="file")
    if not input_file.exists():
        return JsonlLoadResult()

    input_rows = 0
    ok_rows = 0
    error_rows = 0
    stale_rows = 0
    duplicate_ok_rows = 0
    records_by_key: dict[str, AnalysisRecord] = {}
    current_error_keys: set[str] = set()
    try:
        for line_position, row in enumerate(iter_jsonl(input_file)):
            record_index = parse_optional_int(row.get("record_index"))
            effective_index = record_index if record_index is not None else line_position
            if allowed_hashes is None and limit is not None and effective_index >= limit:
                continue

            input_rows += 1
            existing_hash = parse_optional_str(row.get("record_hash"))
            if existing_hash is None and isinstance(row.get("record"), dict):
                existing_hash = record_dict_hash(row["record"])
            record_key = analysis_record_key(
                record_hash_value=existing_hash,
                record_index=record_index,
                allowed_hashes=allowed_hashes,
                allowed_indices=allowed_indices,
            )
            if record_key is None:
                stale_rows += 1
                continue
            if row.get("status") != "ok":
                records_by_key.pop(record_key, None)
                current_error_keys.add(record_key)
                continue
            ok_rows += 1
            current_error_keys.discard(record_key)
            normalized_record_index = (
                current_index_by_hash[existing_hash]
                if existing_hash is not None
                and current_index_by_hash is not None
                and existing_hash in current_index_by_hash
                else record_index
            )

            section = row.get(section_name)
            if not isinstance(section, dict):
                record = AnalysisRecord(record_index=normalized_record_index, record_hash=existing_hash)
                if record_key in records_by_key:
                    duplicate_ok_rows += 1
                records_by_key[record_key] = record
                continue

            record = AnalysisRecord(
                record_index=normalized_record_index,
                record_hash=existing_hash,
                labels=tuple(extract_labels(section, min_confidence=min_confidence)),
                dominant_category=parse_optional_str(section.get("dominant_category")),
                sentiment_polarity=parse_optional_str(section.get("sentiment_polarity")),
            )
            if record_key in records_by_key:
                duplicate_ok_rows += 1
            records_by_key[record_key] = record
    except Exception as exc:  # noqa: BLE001
        error_rows = len(current_error_keys)
        input_status[input_name].update({"status": "error", "error": str(exc)})
        warnings.append(f"Failed to load {input_name}: {exc}")
        return JsonlLoadResult(
            input_rows=input_rows,
            ok_rows=ok_rows,
            error_rows=error_rows,
            stale_rows=stale_rows,
            duplicate_ok_rows=duplicate_ok_rows,
            records=list(records_by_key.values()),
        )

    error_rows = len(current_error_keys)
    records = list(records_by_key.values())
    input_status[input_name].update(
        {
            "status": "loaded",
            "input_rows": input_rows,
            "ok_rows": ok_rows,
            "error_rows": error_rows,
            "stale_rows": stale_rows,
            "duplicate_ok_rows": duplicate_ok_rows,
            "records": len(records),
        }
    )
    return JsonlLoadResult(
        input_rows=input_rows,
        ok_rows=ok_rows,
        error_rows=error_rows,
        stale_rows=stale_rows,
        duplicate_ok_rows=duplicate_ok_rows,
        records=records,
    )


def iter_jsonl(input_file: Path) -> Iterable[dict[str, Any]]:
    with input_file.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if line.strip():
                yield json.loads(line)


def analysis_record_key(
    *,
    record_hash_value: str | None,
    record_index: int | None,
    allowed_hashes: set[str] | None,
    allowed_indices: set[int] | None,
) -> str | None:
    if record_hash_value:
        if allowed_hashes is not None and record_hash_value not in allowed_hashes:
            return None
        return f"hash:{record_hash_value}"
    if record_index is not None:
        if allowed_indices is not None and record_index not in allowed_indices:
            return None
        return f"index:{record_index}"
    return None


def extract_labels(section: dict[str, Any], *, min_confidence: float) -> list[Label]:
    raw_labels = section.get("labels", [])
    if not isinstance(raw_labels, list):
        return []

    selected: dict[str, float] = {}
    for raw_label in raw_labels:
        if not isinstance(raw_label, dict):
            continue
        category = raw_label.get("category", raw_label.get("label"))
        confidence = raw_label.get("confidence", 1.0)
        if not isinstance(category, str):
            continue
        if not isinstance(confidence, int | float) or confidence < min_confidence:
            continue
        selected[category] = max(float(confidence), selected.get(category, 0.0))

    return [Label(category=category, confidence=confidence) for category, confidence in sorted(selected.items())]


def load_topic_file(
    topic_file: Path,
    topic_info_file: Path,
    *,
    limit: int | None,
    input_status: dict[str, dict[str, Any]],
    warnings: list[str],
) -> TopicLoadResult:
    input_status["topic_file"] = describe_path(topic_file, expected_kind="file")
    input_status["topic_info_file"] = describe_path(topic_info_file, expected_kind="file")
    if not topic_file.exists():
        return TopicLoadResult()

    topic_names: dict[str, str] = {}
    if topic_info_file.exists():
        try:
            topic_names = load_topic_names(topic_info_file)
            input_status["topic_info_file"].update({"status": "loaded", "topics": len(topic_names)})
        except Exception as exc:  # noqa: BLE001 - topic info is optional metadata for labeling topics.
            input_status["topic_info_file"].update({"status": "error", "error": str(exc)})
            warnings.append(f"Failed to load topic info file: {exc}")

    input_rows = 0
    topic_rows = 0
    noise_rows = 0
    records: list[TopicRecord] = []
    try:
        with topic_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                record_index = parse_optional_int(row.get("record_index"))
                if record_index is None:
                    continue
                if limit is not None and record_index >= limit:
                    continue

                input_rows += 1
                topic = parse_optional_str(row.get("topic"))
                if topic is None:
                    continue
                if topic == NOISE_TOPIC:
                    noise_rows += 1
                    continue

                topic_rows += 1
                records.append(
                    TopicRecord(
                        record_index=record_index,
                        topic=topic,
                        topic_label=topic_names.get(topic, f"Topic {topic}"),
                        probability=parse_float(row.get("topic_probability")),
                    )
                )
    except Exception as exc:  # noqa: BLE001
        input_status["topic_file"].update({"status": "error", "error": str(exc)})
        warnings.append(f"Failed to load topic file: {exc}")
        return TopicLoadResult(input_rows=input_rows, topic_rows=topic_rows, noise_rows=noise_rows, records=records)

    input_status["topic_file"].update(
        {
            "status": "loaded",
            "input_rows": input_rows,
            "topic_rows": topic_rows,
            "noise_rows": noise_rows,
            "records": len(records),
        }
    )
    return TopicLoadResult(input_rows=input_rows, topic_rows=topic_rows, noise_rows=noise_rows, records=records)


def load_topic_names(topic_info_file: Path) -> dict[str, str]:
    if not topic_info_file.exists():
        return {}

    topic_names: dict[str, str] = {}
    with topic_info_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            topic = parse_optional_str(row.get("Topic", row.get("topic")))
            if topic is None:
                continue
            name = parse_optional_str(row.get("Name", row.get("name")))
            keywords = parse_optional_str(row.get("Keywords", row.get("Representation")))
            if name and keywords:
                topic_names[topic] = f"Topic {topic}: {name} ({keywords})"
            elif name:
                topic_names[topic] = f"Topic {topic}: {name}"
            else:
                topic_names[topic] = f"Topic {topic}"
    return topic_names


def describe_path(path: Path, *, expected_kind: str) -> dict[str, Any]:
    if expected_kind == "dir":
        exists = path.is_dir()
    else:
        exists = path.is_file()
    return {
        "path": str(path),
        "status": "present" if exists else "missing",
    }


def summarize_labels(records: list[AnalysisRecord], *, denominator: int) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    record_counts: Counter[str] = Counter()
    confidences: dict[str, list[float]] = defaultdict(list)
    for record in records:
        categories_in_record: set[str] = set()
        for label in record.labels:
            counts[label.category] += 1
            categories_in_record.add(label.category)
            confidences[label.category].append(label.confidence)
        for category in categories_in_record:
            record_counts[category] += 1

    rows = []
    for category, count in counts.most_common():
        values = confidences[category]
        rows.append(
            {
                "category": category,
                "count": count,
                "record_count": record_counts[category],
                "share_of_records": safe_divide(record_counts[category], denominator),
                "mean_confidence": sum(values) / len(values) if values else 0.0,
                "min_confidence": min(values) if values else 0.0,
                "max_confidence": max(values) if values else 0.0,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "category",
            "count",
            "record_count",
            "share_of_records",
            "mean_confidence",
            "min_confidence",
            "max_confidence",
        ],
    )


def summarize_single_field(values: Iterable[str | None], *, field_name: str, denominator: int) -> pd.DataFrame:
    counts = Counter(value for value in values if value)
    rows = [
        {field_name: value, "count": count, "share_of_records": safe_divide(count, denominator)}
        for value, count in counts.most_common()
    ]
    return pd.DataFrame(rows, columns=[field_name, "count", "share_of_records"])


def summarize_topics(records: list[TopicRecord]) -> pd.DataFrame:
    counts: Counter[tuple[str, str]] = Counter()
    probabilities: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        key = (record.topic, record.topic_label)
        counts[key] += 1
        if record.probability is not None:
            probabilities[key].append(record.probability)

    total = sum(counts.values())
    rows = []
    for (topic, label), count in counts.most_common():
        values = probabilities[(topic, label)]
        rows.append(
            {
                "topic": topic,
                "topic_label": label,
                "count": count,
                "share_of_topic_records": safe_divide(count, total),
                "mean_probability": sum(values) / len(values) if values else "",
            }
        )
    return pd.DataFrame(rows, columns=["topic", "topic_label", "count", "share_of_topic_records", "mean_probability"])


def build_analysis_crosstab(
    pairs: list[tuple[AnalysisRecord, AnalysisRecord]],
    *,
    left_name: str,
    right_values,
) -> pd.DataFrame:
    matrix: Counter[tuple[str, str]] = Counter()
    row_counts: Counter[str] = Counter()
    column_counts: Counter[str] = Counter()
    for pair in pairs:
        left_record = pair[0]
        left_values = [label.category for label in left_record.labels]
        values = [value for value in right_values(pair) if value]
        for left_value in left_values:
            row_counts[left_value] += 1
            for right_value in values:
                matrix[(left_value, str(right_value))] += 1
                column_counts[str(right_value)] += 1

    return matrix_to_dataframe(
        matrix,
        row_labels=[label for label, _ in row_counts.most_common()],
        column_labels=[label for label, _ in column_counts.most_common()],
        row_name=left_name,
    )


def build_topic_crosstab(pairs: list[tuple[TopicRecord, AnalysisRecord]]) -> pd.DataFrame:
    matrix: Counter[tuple[str, str]] = Counter()
    row_labels_by_topic: dict[str, str] = {}
    row_counts: Counter[str] = Counter()
    column_counts: Counter[str] = Counter()
    for topic_record, analysis_record in pairs:
        row_key = topic_record.topic
        row_labels_by_topic[row_key] = topic_record.topic_label
        for label in analysis_record.labels:
            matrix[(row_key, label.category)] += 1
            row_counts[row_key] += 1
            column_counts[label.category] += 1

    rows = []
    columns = [label for label, _ in column_counts.most_common()]
    for topic, _ in row_counts.most_common():
        row = {"topic": topic, "topic_label": row_labels_by_topic.get(topic, f"Topic {topic}")}
        for column in columns:
            row[column] = matrix[(topic, column)]
        rows.append(row)

    return pd.DataFrame(rows, columns=["topic", "topic_label", *columns])


def matrix_to_dataframe(
    matrix: Counter[tuple[str, str]],
    *,
    row_labels: list[str],
    column_labels: list[str],
    row_name: str,
) -> pd.DataFrame:
    rows = []
    for row_label in row_labels:
        row = {row_name: row_label}
        for column_label in column_labels:
            row[column_label] = matrix[(row_label, column_label)]
        rows.append(row)
    return pd.DataFrame(rows, columns=[row_name, *column_labels])


def match_analysis_records(
    left_records: list[AnalysisRecord],
    right_records: list[AnalysisRecord],
) -> list[tuple[AnalysisRecord, AnalysisRecord]]:
    right_by_hash = {record.record_hash: record for record in right_records if record.record_hash}
    right_by_index = records_by_index(right_records)
    matched: list[tuple[AnalysisRecord, AnalysisRecord]] = []
    seen_right_keys: set[str] = set()

    for left in left_records:
        right: AnalysisRecord | None = None
        right_key: str | None = None
        if left.record_hash and left.record_hash in right_by_hash:
            right = right_by_hash[left.record_hash]
            right_key = f"hash:{left.record_hash}"
        elif left.record_index is not None and left.record_index in right_by_index:
            right = right_by_index[left.record_index]
            right_key = f"index:{left.record_index}"

        if right is not None and right_key not in seen_right_keys:
            matched.append((left, right))
            if right_key is not None:
                seen_right_keys.add(right_key)

    return matched


def records_by_index(records: Iterable[AnalysisRecord]) -> dict[int, AnalysisRecord]:
    return {record.record_index: record for record in records if record.record_index is not None}


def build_word_frequency(records: list[Any], stopwords: set[str], *, warnings: list[str]) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for record in records:
        for token in jieba.lcut(record.content):
            normalized = token.strip()
            if not normalized:
                continue
            if normalized in stopwords or normalized.lower() in stopwords:
                continue
            if len(normalized) < 2:
                continue
            if normalized.isdigit() or TOKEN_PUNCTUATION_RE.match(normalized):
                continue
            counter[normalized] += 1

    rows = [{"word": word, "count": count} for word, count in counter.most_common()]
    return pd.DataFrame(rows, columns=["word", "count"])


def load_stopwords(stopwords_file: str | Path | None) -> set[str]:
    stopwords = set(BUILTIN_STOPWORDS)
    if stopwords_file is None:
        return stopwords

    path = Path(stopwords_file)
    if not path.exists():
        return stopwords
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if word:
            stopwords.add(word)
    return stopwords


def build_quality_metrics(
    *,
    cleaned_records: list[Any],
    content_data: JsonlLoadResult,
    sentiment_data: JsonlLoadResult,
    topic_data: TopicLoadResult,
    content_records: list[AnalysisRecord],
    sentiment_records: list[AnalysisRecord],
    matched_content_sentiment: list[tuple[AnalysisRecord, AnalysisRecord]],
    matched_topic_content: list[tuple[TopicRecord, AnalysisRecord]],
    matched_topic_sentiment: list[tuple[TopicRecord, AnalysisRecord]],
) -> dict[str, int | float]:
    content_fallback_records = sum(
        1 for record in content_records if any(label.category == CONTENT_FALLBACK_CATEGORY for label in record.labels)
    )
    sentiment_fallback_records = sum(
        1 for record in sentiment_records if any(label.category == SENTIMENT_FALLBACK_CATEGORY for label in record.labels)
    )
    return {
        "cleaned_records": len(cleaned_records),
        "content_input_rows": content_data.input_rows,
        "content_ok_rows": content_data.ok_rows,
        "content_error_rows": content_data.error_rows,
        "content_stale_rows": content_data.stale_rows,
        "content_duplicate_ok_rows": content_data.duplicate_ok_rows,
        "content_records_with_labels": sum(1 for record in content_records if record.labels),
        "content_label_occurrences": sum(len(record.labels) for record in content_records),
        "content_fallback_records": content_fallback_records,
        "sentiment_input_rows": sentiment_data.input_rows,
        "sentiment_ok_rows": sentiment_data.ok_rows,
        "sentiment_error_rows": sentiment_data.error_rows,
        "sentiment_stale_rows": sentiment_data.stale_rows,
        "sentiment_duplicate_ok_rows": sentiment_data.duplicate_ok_rows,
        "sentiment_records_with_labels": sum(1 for record in sentiment_records if record.labels),
        "sentiment_label_occurrences": sum(len(record.labels) for record in sentiment_records),
        "sentiment_fallback_records": sentiment_fallback_records,
        "matched_content_sentiment_records": len(matched_content_sentiment),
        "topic_input_rows": topic_data.input_rows,
        "topic_rows": topic_data.topic_rows,
        "topic_noise_rows": topic_data.noise_rows,
        "matched_topic_content_records": len(matched_topic_content),
        "matched_topic_sentiment_records": len(matched_topic_sentiment),
    }


def write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / name
        table.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = path
    return paths


def write_figures(
    *,
    tables: dict[str, pd.DataFrame],
    cooccurrence_dir: Path,
    figures_dir: Path,
    top_n: int,
    font_path: Path | None,
    warnings: list[str],
) -> dict[str, Path]:
    figures: dict[str, Path] = {}
    figures["content_label_distribution.png"] = figures_dir / "content_label_distribution.png"
    plot_bar(
        tables.get("content_label_summary.csv"),
        label_col="category",
        value_col="count",
        title="Content Label Distribution",
        output_file=figures["content_label_distribution.png"],
        top_n=top_n,
    )

    figures["sentiment_label_distribution.png"] = figures_dir / "sentiment_label_distribution.png"
    plot_bar(
        tables.get("sentiment_label_summary.csv"),
        label_col="category",
        value_col="count",
        title="Sentiment Label Distribution",
        output_file=figures["sentiment_label_distribution.png"],
        top_n=top_n,
    )

    figures["sentiment_polarity_distribution.png"] = figures_dir / "sentiment_polarity_distribution.png"
    plot_bar(
        tables.get("sentiment_polarity_summary.csv"),
        label_col="polarity",
        value_col="count",
        title="Sentiment Polarity Distribution",
        output_file=figures["sentiment_polarity_distribution.png"],
        top_n=top_n,
    )

    figures["content_sentiment_heatmap.png"] = figures_dir / "content_sentiment_heatmap.png"
    plot_heatmap(
        tables.get("content_sentiment_crosstab.csv"),
        row_col="content_label",
        title="Content x Sentiment Heatmap",
        output_file=figures["content_sentiment_heatmap.png"],
        top_n=top_n,
    )

    figures["topic_distribution.png"] = figures_dir / "topic_distribution.png"
    plot_bar(
        tables.get("topic_summary.csv"),
        label_col="topic_label",
        value_col="count",
        title="Topic Distribution",
        output_file=figures["topic_distribution.png"],
        top_n=top_n,
    )

    figures["wordcloud.png"] = figures_dir / "wordcloud.png"
    plot_wordcloud(
        tables.get("word_frequency.csv"),
        output_file=figures["wordcloud.png"],
        font_path=font_path,
        warnings=warnings,
    )

    figures["content_sentiment_network.png"] = figures_dir / "content_sentiment_network.png"
    plot_network(
        load_content_sentiment_edges(cooccurrence_dir) or edges_from_crosstab(tables.get("content_sentiment_crosstab.csv"), "content_label"),
        title="Content x Sentiment Co-occurrence Network",
        output_file=figures["content_sentiment_network.png"],
        top_n=top_n,
    )
    return figures


def plot_bar(
    table: pd.DataFrame | None,
    *,
    label_col: str,
    value_col: str,
    title: str,
    output_file: Path,
    top_n: int,
) -> None:
    if table is None or table.empty or label_col not in table or value_col not in table:
        write_placeholder_figure(output_file, title, "No data")
        return

    plot_table = table.sort_values(value_col, ascending=False).head(top_n)
    labels = [shorten_label(str(label), 28) for label in plot_table[label_col].tolist()]
    values = plot_table[value_col].tolist()
    height = max(4, min(12, 0.45 * len(labels) + 2))
    plt.figure(figsize=(10, height))
    plt.barh(labels[::-1], values[::-1], color="#376092")
    plt.title(title)
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()


def plot_heatmap(
    table: pd.DataFrame | None,
    *,
    row_col: str,
    title: str,
    output_file: Path,
    top_n: int,
) -> None:
    if table is None or table.empty or row_col not in table or len(table.columns) <= 1:
        write_placeholder_figure(output_file, title, "No data")
        return

    rows = table.head(top_n)
    value_columns = list(table.columns[1 : top_n + 1])
    matrix = rows[value_columns].fillna(0).astype(float).to_numpy()
    if matrix.size == 0:
        write_placeholder_figure(output_file, title, "No data")
        return

    plt.figure(figsize=(max(8, len(value_columns) * 0.9), max(5, len(rows) * 0.45)))
    plt.imshow(matrix, aspect="auto", cmap="YlGnBu")
    plt.colorbar(label="Count")
    plt.xticks(range(len(value_columns)), [shorten_label(column, 18) for column in value_columns], rotation=45, ha="right")
    plt.yticks(range(len(rows)), [shorten_label(str(label), 24) for label in rows[row_col].tolist()])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()


def plot_wordcloud(
    table: pd.DataFrame | None,
    *,
    output_file: Path,
    font_path: Path | None,
    warnings: list[str],
) -> None:
    if table is None or table.empty or "word" not in table or "count" not in table:
        write_placeholder_figure(output_file, "Word Cloud", "No data")
        return

    frequencies = {str(row["word"]): int(row["count"]) for _, row in table.head(200).iterrows()}
    try:
        wordcloud = WordCloud(
            width=1200,
            height=800,
            background_color="white",
            font_path=str(font_path) if font_path else None,
            max_words=200,
        ).generate_from_frequencies(frequencies)
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_file, dpi=160)
        plt.close()
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Failed to generate wordcloud: {exc}")
        write_placeholder_figure(output_file, "Word Cloud", "Generation failed")


def plot_network(edges: list[tuple[str, str, int]], *, title: str, output_file: Path, top_n: int) -> None:
    if not edges:
        write_placeholder_figure(output_file, title, "No data")
        return

    selected_edges = sorted(edges, key=lambda item: item[2], reverse=True)[: max(top_n * 2, top_n)]
    graph = nx.Graph()
    for source, target, weight in selected_edges:
        graph.add_edge(source, target, weight=weight)

    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(graph, weight="weight", seed=42, k=0.8)
    weights = [max(1.0, math.log(data["weight"] + 1) * 1.8) for _, _, data in graph.edges(data=True)]
    nx.draw_networkx_nodes(graph, pos, node_size=900, node_color="#d8e4bc", edgecolors="#4f6228", linewidths=1.2)
    nx.draw_networkx_edges(graph, pos, width=weights, alpha=0.65, edge_color="#7f7f7f")
    nx.draw_networkx_labels(graph, pos, labels={node: shorten_label(node, 16) for node in graph.nodes}, font_size=9)
    edge_labels = {(source, target): data["weight"] for source, target, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()


def write_placeholder_figure(output_file: Path, title: str, message: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()


def load_content_sentiment_edges(cooccurrence_dir: Path) -> list[tuple[str, str, int]]:
    edge_file = cooccurrence_dir / "content_sentiment_edges.csv"
    if not edge_file.exists():
        return []

    edges: list[tuple[str, str, int]] = []
    with edge_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            source = row.get("source_label") or row.get("source")
            target = row.get("target_label") or row.get("target")
            weight = parse_optional_int(row.get("weight")) or 0
            if source and target and weight > 0:
                edges.append((source, target, weight))
    return edges


def edges_from_crosstab(table: pd.DataFrame | None, row_col: str) -> list[tuple[str, str, int]]:
    if table is None or table.empty or row_col not in table:
        return []

    edges: list[tuple[str, str, int]] = []
    for _, row in table.iterrows():
        source = str(row[row_col])
        for column in table.columns:
            if column == row_col:
                continue
            value = parse_optional_int(row[column])
            if value is not None and value > 0:
                edges.append((source, str(column), value))
    return edges


def write_static_html(
    output_file: Path,
    *,
    tables: dict[str, pd.DataFrame],
    figures: dict[str, Path],
    input_status: dict[str, dict[str, Any]],
    quality_metrics: dict[str, int | float],
) -> None:
    figure_blocks = "\n".join(
        f'<section><h2>{html.escape(path.stem.replace("_", " ").title())}</h2>'
        f'<img src="{html.escape(str(path.relative_to(output_file.parent)).replace("\\\\", "/"))}" alt="{html.escape(path.stem)}"></section>'
        for path in figures.values()
    )
    status_table = dict_to_html_table(input_status)
    quality_table = pd.DataFrame([{"metric": key, "value": value} for key, value in quality_metrics.items()]).to_html(index=False)
    table_blocks = "\n".join(
        f"<section><h2>{html.escape(name)}</h2>{table.head(20).to_html(index=False)}</section>"
        for name, table in tables.items()
        if not table.empty
    )
    output_file.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>AIGC Comment Analysis Report</title>
  <style>
    body {{ font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif; margin: 32px; color: #222; }}
    h1 {{ margin-bottom: 4px; }}
    section {{ margin: 28px 0; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f2f2f2; }}
  </style>
</head>
<body>
  <h1>AIGC Comment Analysis Report</h1>
  <p>Generated at {html.escape(datetime.now(timezone.utc).isoformat())}</p>
  <section><h2>Input Status</h2>{status_table}</section>
  <section><h2>Data Quality</h2>{quality_table}</section>
  {figure_blocks}
  {table_blocks}
</body>
</html>
""",
        encoding="utf-8",
    )


def write_interactive_html(
    output_file: Path,
    *,
    tables: dict[str, pd.DataFrame],
    cooccurrence_dir: Path,
    top_n: int,
    warnings: list[str],
) -> None:
    try:
        page = Page(page_title="AIGC Comment Interactive Report", layout=Page.SimplePageLayout)
        add_bar_chart(page, tables.get("content_label_summary.csv"), "category", "count", "Content Labels", top_n)
        add_bar_chart(page, tables.get("sentiment_label_summary.csv"), "category", "count", "Sentiment Labels", top_n)
        add_pie_chart(page, tables.get("sentiment_polarity_summary.csv"), "polarity", "count", "Sentiment Polarity", top_n)
        add_heatmap_chart(page, tables.get("content_sentiment_crosstab.csv"), "content_label", "Content x Sentiment", top_n)
        add_network_chart(
            page,
            load_content_sentiment_edges(cooccurrence_dir)
            or edges_from_crosstab(tables.get("content_sentiment_crosstab.csv"), "content_label"),
            "Content x Sentiment Network",
            top_n,
        )
        page.render(str(output_file))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Failed to generate interactive report: {exc}")
        output_file.write_text(
            "<!doctype html><meta charset='utf-8'><title>Interactive Report</title><p>Interactive report generation failed.</p>",
            encoding="utf-8",
        )


def add_bar_chart(page: Page, table: pd.DataFrame | None, label_col: str, value_col: str, title: str, top_n: int) -> None:
    if table is None or table.empty or label_col not in table or value_col not in table:
        return
    rows = table.sort_values(value_col, ascending=False).head(top_n)
    chart = (
        Bar()
        .add_xaxis([str(value) for value in rows[label_col].tolist()])
        .add_yaxis("count", [int(value) for value in rows[value_col].tolist()])
        .set_global_opts(title_opts=opts.TitleOpts(title=title), xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=35)))
    )
    page.add(chart)


def add_pie_chart(page: Page, table: pd.DataFrame | None, label_col: str, value_col: str, title: str, top_n: int) -> None:
    if table is None or table.empty or label_col not in table or value_col not in table:
        return
    rows = table.sort_values(value_col, ascending=False).head(top_n)
    chart = (
        Pie()
        .add("", [(str(row[label_col]), int(row[value_col])) for _, row in rows.iterrows()])
        .set_global_opts(title_opts=opts.TitleOpts(title=title))
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
    )
    page.add(chart)


def add_heatmap_chart(page: Page, table: pd.DataFrame | None, row_col: str, title: str, top_n: int) -> None:
    if table is None or table.empty or row_col not in table or len(table.columns) <= 1:
        return
    rows = table.head(top_n)
    columns = list(table.columns[1 : top_n + 1])
    data = []
    for y_index, (_, row) in enumerate(rows.iterrows()):
        for x_index, column in enumerate(columns):
            data.append([x_index, y_index, int(row[column])])
    chart = (
        HeatMap()
        .add_xaxis(columns)
        .add_yaxis("count", [str(value) for value in rows[row_col].tolist()], data)
        .set_global_opts(title_opts=opts.TitleOpts(title=title), visualmap_opts=opts.VisualMapOpts())
    )
    page.add(chart)


def add_network_chart(page: Page, edges: list[tuple[str, str, int]], title: str, top_n: int) -> None:
    if not edges:
        return
    selected_edges = sorted(edges, key=lambda item: item[2], reverse=True)[: max(top_n * 2, top_n)]
    nodes = sorted({source for source, _, _ in selected_edges} | {target for _, target, _ in selected_edges})
    chart = (
        PyeGraph()
        .add(
            "",
            nodes=[{"name": node, "symbolSize": 18} for node in nodes],
            links=[{"source": source, "target": target, "value": weight} for source, target, weight in selected_edges],
            repulsion=1200,
            edge_label=opts.LabelOpts(is_show=True, formatter="{c}"),
        )
        .set_global_opts(title_opts=opts.TitleOpts(title=title))
    )
    page.add(chart)


def dict_to_html_table(data: dict[str, dict[str, Any]]) -> str:
    rows = []
    for name, values in data.items():
        row_values = html.escape(json.dumps(values, ensure_ascii=False))
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{row_values}</td></tr>")
    return "<table><thead><tr><th>Input</th><th>Status</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def configure_chinese_font(warnings: list[str]) -> tuple[Path | None, str | None]:
    preferred_names = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimSun"]
    preferred_files = ["msyh.ttc", "simhei.ttf", "simsun.ttc"]
    for font_path in font_manager.findSystemFonts():
        path = Path(font_path)
        try:
            font_name = font_manager.FontProperties(fname=font_path).get_name()
        except RuntimeError:
            continue
        if font_name in preferred_names or path.name.lower() in preferred_files:
            plt.rcParams["font.sans-serif"] = [font_name, *plt.rcParams.get("font.sans-serif", [])]
            plt.rcParams["axes.unicode_minus"] = False
            return path, font_name

    plt.rcParams["axes.unicode_minus"] = False
    warnings.append("No preferred Chinese font found. Figures may render Chinese labels with missing glyphs.")
    return None, None


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def shorten_label(label: str, max_length: int) -> str:
    return label if len(label) <= max_length else f"{label[: max_length - 1]}…"


def parse_optional_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def parse_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def parse_optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build summary tables and visual reports for comment analysis.")
    parser.add_argument("--cleaned-file", default=str(DEFAULT_CLEANED_FILE))
    parser.add_argument("--content-file", default=str(DEFAULT_CONTENT_FILE))
    parser.add_argument("--sentiment-file", default=str(DEFAULT_SENTIMENT_FILE))
    parser.add_argument("--topic-file", default=str(DEFAULT_TOPIC_FILE))
    parser.add_argument("--topic-info-file", default=str(DEFAULT_TOPIC_INFO_FILE))
    parser.add_argument("--cooccurrence-dir", default=str(DEFAULT_COOCCURRENCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-sentiment-confidence", type=float, default=None)
    parser.add_argument("--stopwords-file", default=None)
    args = parser.parse_args()

    report = build_analysis_report(
        cleaned_file=args.cleaned_file,
        content_file=args.content_file,
        sentiment_file=args.sentiment_file,
        topic_file=args.topic_file,
        topic_info_file=args.topic_info_file,
        cooccurrence_dir=args.cooccurrence_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        top_n=args.top_n,
        min_confidence=args.min_confidence,
        min_sentiment_confidence=args.min_sentiment_confidence,
        stopwords_file=args.stopwords_file,
    )
    print(f"Output dir: {report.output_dir}")
    print(f"Tables: {len(report.tables)}")
    print(f"Figures: {len(report.figures)}")
    print(f"Report HTML: {report.static_html}")
    print(f"Interactive HTML: {report.interactive_html}")
    print(f"Metadata: {report.metadata_file}")


if __name__ == "__main__":
    main()
