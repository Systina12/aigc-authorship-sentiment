from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import jieba
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from pandas.errors import EmptyDataError
from wordcloud import WordCloud

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloader import CommentRecord, DataLoader
from scripts.analyze_content import record_dict_hash, record_hash
from scripts.cluster_topics import (
    COMMENT_TOPICS_FILE_NAME,
    DEFAULT_EMBEDDING_MODEL,
    METADATA_FILE_NAME,
    REPRESENTATIVE_DOCS_FILE_NAME,
    TOPIC_INFO_FILE_NAME,
    TopicDocument,
    TopicModel,
    create_topic_model,
    normalize_representative_docs,
    normalize_topic_info,
    write_comment_topics,
    write_metadata as write_topic_metadata,
    write_representative_docs,
    write_topic_info,
)


DEFAULT_CLEANED_FILE = Path("data/cleaned/comments_cleaned.csv")
DEFAULT_CONTENT_FILE = Path("data/content_analysis/comment_labels.jsonl")
DEFAULT_SENTIMENT_FILE = Path("data/sentiment_analysis/comment_sentiment.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/repaired_analysis")
DEFAULT_MIN_TOPIC_TEXT_LENGTH = 8
DEFAULT_MIN_TOPIC_SIZE = 50
DISPLAY_MULTIPLIER = 10

CONTENT_FALLBACK_CATEGORY = "无法归类/无关讨论"
SENTIMENT_FALLBACK_CATEGORY = "中性/无法判断"
NEGATIVE_POLARITY = "negative"
NEUTRAL_POLARITY = "neutral"

SENTIMENT_CATEGORY_ALIASES = {
    "失望": ("焦虑", NEGATIVE_POLARITY),
    "悲观": ("焦虑", NEGATIVE_POLARITY),
    "无奈": ("焦虑", NEGATIVE_POLARITY),
    "遗憾": ("焦虑", NEGATIVE_POLARITY),
    "厌恶": ("愤怒", NEGATIVE_POLARITY),
    "惊讶": (SENTIMENT_FALLBACK_CATEGORY, NEUTRAL_POLARITY),
}

NOISE_TOKENS = {
    "回复",
    "偷笑",
    "飞吻",
    "关注",
    "害羞",
    "消息",
    "滴滴",
    "哈哈",
    "哈哈哈",
    "哈哈哈哈",
    "谢谢",
    "微笑",
    "石化",
    "玫瑰",
    "皱眉",
    "捂脸",
    "流汗",
    "笑哭",
    "哭惹",
    "赞赞",
    "宝子",
    "宝子发",
    "求分享",
    "分享",
    "哪里下载",
    "怎么下载",
    "安装包",
    "求安装包",
    "关注了",
    "已关注",
    "求资料",
    "私你",
    "私信",
    "我下",
    "想要",
}
STOPWORDS = {
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
    "一下",
    "出来",
    "不会",
    "知道",
    "感觉",
    "需要",
    "看看",
    "这种",
    "东西",
    "这样",
    "一样",
    "还有",
    "只是",
    "直接",
    "人工智能",
    "ai",
    "AI",
    *NOISE_TOKENS,
}
TOKEN_PUNCTUATION_RE = re.compile(r"^[\W_]+$", re.UNICODE)
UNSUPPORTED_SENTIMENT_RE = re.compile(r"unsupported sentiment category: '([^']+)'")


@dataclass(frozen=True)
class Label:
    category: str
    confidence: float


@dataclass
class RepairedRecord:
    record_index: int
    record_hash: str
    record: CommentRecord
    content_labels: list[Label]
    sentiment_labels: list[Label]
    dominant_sentiment: str
    sentiment_polarity: str
    content_status: str
    sentiment_status: str
    content_repaired: bool = False
    sentiment_repaired: bool = False
    is_interaction_noise: bool = False
    normalized_text: str = ""
    in_core_opinion: bool = False
    in_core_sentiment: bool = False
    in_topic_input: bool = False


@dataclass(frozen=True)
class RepairedAnalysisReport:
    output_dir: Path
    tables_dir: Path
    figures_dir: Path
    report_file: Path
    interactive_report_file: Path
    metadata_file: Path
    repair_decisions_file: Path
    topic_dir: Path | None


def build_repaired_analysis(
    *,
    cleaned_file: str | Path = DEFAULT_CLEANED_FILE,
    content_file: str | Path = DEFAULT_CONTENT_FILE,
    sentiment_file: str | Path = DEFAULT_SENTIMENT_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    min_confidence: float = 0.0,
    min_sentiment_confidence: float = 0.0,
    min_topic_text_length: int = DEFAULT_MIN_TOPIC_TEXT_LENGTH,
    min_topic_size: int = DEFAULT_MIN_TOPIC_SIZE,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    top_n: int = 20,
    skip_topic_clustering: bool = False,
    display_multiplier: int = DISPLAY_MULTIPLIER,
    topic_model: TopicModel | None = None,
) -> RepairedAnalysisReport:
    validate_parameters(
        min_confidence=min_confidence,
        min_sentiment_confidence=min_sentiment_confidence,
        min_topic_text_length=min_topic_text_length,
        min_topic_size=min_topic_size,
        top_n=top_n,
        display_multiplier=display_multiplier,
    )
    output_path = Path(output_dir)
    tables_dir = output_path / "tables"
    figures_dir = output_path / "figures"
    report_dir = output_path / "report"
    topic_dir = output_path / "topic_clustering"
    for path in (tables_dir, figures_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)
    cleanup_outputs(output_path)

    records = DataLoader().load(cleaned_file)
    hash_by_index = {index: record_hash(record) for index, record in enumerate(records)}
    record_by_hash = {hash_by_index[index]: record for index, record in enumerate(records)}
    repair_decisions: list[dict[str, Any]] = []
    content_entries = load_content_entries(
        Path(content_file),
        allowed_hashes=set(record_by_hash),
        min_confidence=min_confidence,
        repair_decisions=repair_decisions,
    )
    sentiment_entries = load_sentiment_entries(
        Path(sentiment_file),
        allowed_hashes=set(record_by_hash),
        min_confidence=min_sentiment_confidence,
        repair_decisions=repair_decisions,
    )

    repaired_records = build_repaired_records(
        records=records,
        hash_by_index=hash_by_index,
        content_entries=content_entries,
        sentiment_entries=sentiment_entries,
        min_topic_text_length=min_topic_text_length,
    )

    write_analysis_records(repaired_records, tables_dir / "analysis_records.csv")
    write_repair_decisions(repair_decisions, output_path / "repair_decisions.jsonl")
    table_paths = write_summary_tables(
        repaired_records=repaired_records,
        tables_dir=tables_dir,
        display_multiplier=display_multiplier,
    )

    topic_summary_rows: list[dict[str, Any]] = []
    if not skip_topic_clustering:
        topic_summary_rows = run_topic_clustering(
            repaired_records=repaired_records,
            topic_dir=topic_dir,
            embedding_model=embedding_model,
            min_topic_size=min_topic_size,
            min_topic_text_length=min_topic_text_length,
            topic_model=topic_model,
        )
        write_topic_summary(topic_summary_rows, tables_dir / "core_topic_summary.csv", display_multiplier)
        table_paths["core_topic_summary.csv"] = tables_dir / "core_topic_summary.csv"
    else:
        topic_dir = None

    figure_paths = write_figures(
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        topic_summary_rows=topic_summary_rows,
        top_n=top_n,
    )
    metadata = build_metadata(
        repaired_records=repaired_records,
        repair_decisions=repair_decisions,
        cleaned_file=Path(cleaned_file),
        content_file=Path(content_file),
        sentiment_file=Path(sentiment_file),
        output_dir=output_path,
        display_multiplier=display_multiplier,
        topic_summary_rows=topic_summary_rows,
        skip_topic_clustering=skip_topic_clustering,
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        min_topic_text_length=min_topic_text_length,
        table_paths=table_paths,
        figure_paths=figure_paths,
    )
    metadata_file = output_path / "repair_metadata.json"
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report_file = report_dir / "report.html"
    interactive_report_file = report_dir / "interactive_report.html"
    write_html_report(
        report_file=report_file,
        interactive_report_file=interactive_report_file,
        figures=figure_paths,
        metadata=metadata,
    )

    return RepairedAnalysisReport(
        output_dir=output_path,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        report_file=report_file,
        interactive_report_file=interactive_report_file,
        metadata_file=metadata_file,
        repair_decisions_file=output_path / "repair_decisions.jsonl",
        topic_dir=topic_dir,
    )


def validate_parameters(
    *,
    min_confidence: float,
    min_sentiment_confidence: float,
    min_topic_text_length: int,
    min_topic_size: int,
    top_n: int,
    display_multiplier: int,
) -> None:
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0.0 and 1.0")
    if not 0 <= min_sentiment_confidence <= 1:
        raise ValueError("min_sentiment_confidence must be between 0.0 and 1.0")
    if min_topic_text_length < 1:
        raise ValueError("min_topic_text_length must be greater than or equal to 1")
    if min_topic_size < 2:
        raise ValueError("min_topic_size must be greater than or equal to 2")
    if top_n < 1:
        raise ValueError("top_n must be greater than or equal to 1")
    if display_multiplier < 1:
        raise ValueError("display_multiplier must be greater than or equal to 1")


def cleanup_outputs(output_dir: Path) -> None:
    legacy_suffix = f"x{DISPLAY_MULTIPLIER}"
    managed_files = [
        "repair_metadata.json",
        "repair_decisions.jsonl",
        "tables/analysis_records.csv",
        "tables/full_content_label_summary.csv",
        "tables/full_sentiment_label_summary.csv",
        "tables/core_content_label_summary.csv",
        "tables/core_sentiment_label_summary.csv",
        "tables/core_polarity_summary.csv",
        "tables/core_content_sentiment_crosstab.csv",
        "tables/core_content_polarity_crosstab.csv",
        "tables/core_word_frequency.csv",
        "tables/core_topic_summary.csv",
        "tables/data_quality_summary.csv",
        "figures/core_content_label_distribution.png",
        "figures/core_sentiment_label_distribution.png",
        "figures/core_polarity_distribution.png",
        "figures/core_content_sentiment_crosstab_heatmap.png",
        "figures/core_content_polarity_crosstab_heatmap.png",
        "figures/core_wordcloud.png",
        "figures/core_topic_distribution.png",
        "figures/data_quality_summary_table.png",
        "figures/full_content_label_summary_table.png",
        "figures/full_sentiment_label_summary_table.png",
        "figures/core_content_label_summary_table.png",
        "figures/core_sentiment_label_summary_table.png",
        "figures/core_polarity_summary_table.png",
        "figures/core_content_sentiment_crosstab_table.png",
        "figures/core_content_polarity_crosstab_table.png",
        "figures/core_topic_summary_table.png",
        "figures/core_word_frequency_table.png",
        f"figures/core_content_label_distribution_{legacy_suffix}.png",
        f"figures/core_sentiment_label_distribution_{legacy_suffix}.png",
        f"figures/core_polarity_distribution_{legacy_suffix}.png",
        f"figures/core_wordcloud_{legacy_suffix}.png",
        f"figures/core_topic_distribution_{legacy_suffix}.png",
        f"topic_clustering/{COMMENT_TOPICS_FILE_NAME}",
        f"topic_clustering/{TOPIC_INFO_FILE_NAME}",
        f"topic_clustering/{REPRESENTATIVE_DOCS_FILE_NAME}",
        f"topic_clustering/{METADATA_FILE_NAME}",
        "report/report.html",
        "report/interactive_report.html",
    ]
    for relative in managed_files:
        path = output_dir / relative
        if path.is_file():
            path.unlink()


def load_content_entries(
    input_file: Path,
    *,
    allowed_hashes: set[str],
    min_confidence: float,
    repair_decisions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    if not input_file.exists():
        return entries
    for row in iter_jsonl(input_file):
        record_hash_value = row_hash(row)
        if record_hash_value not in allowed_hashes:
            continue
        if row.get("status") != "ok":
            put_preferred_entry(entries, record_hash_value, {"status": "error", "labels": []})
            continue
        section = row.get("analysis")
        labels = extract_labels(section if isinstance(section, dict) else {}, min_confidence=min_confidence)
        repaired = False
        if not labels:
            labels = [Label(CONTENT_FALLBACK_CATEGORY, 0.5)]
            repaired = True
        entry = {"status": "ok", "labels": labels, "repaired": repaired}
        if repaired:
            entry["repair_decision"] = repair_decision(
                row,
                repair_type="content_empty_labels",
                mapped_category=CONTENT_FALLBACK_CATEGORY,
            )
        put_preferred_entry(entries, record_hash_value, entry)
    repair_decisions.extend(collect_repair_decisions(entries))
    return entries


def load_sentiment_entries(
    input_file: Path,
    *,
    allowed_hashes: set[str],
    min_confidence: float,
    repair_decisions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    if not input_file.exists():
        return entries
    for row in iter_jsonl(input_file):
        record_hash_value = row_hash(row)
        if record_hash_value not in allowed_hashes:
            continue
        if row.get("status") != "ok":
            alias = extract_unsupported_sentiment_alias(str(row.get("error_message", "")))
            if alias is None:
                put_preferred_entry(entries, record_hash_value, {"status": "error", "labels": []})
                continue
            mapped_category, polarity = alias
            label = Label(mapped_category, 0.5)
            put_preferred_entry(
                entries,
                record_hash_value,
                {
                    "status": "ok",
                    "labels": [label],
                    "dominant": mapped_category,
                    "polarity": polarity,
                    "repaired": True,
                    "repair_decision": repair_decision(
                        row,
                        repair_type="sentiment_category_alias",
                        mapped_category=mapped_category,
                    ),
                },
            )
            continue

        section = row.get("sentiment")
        labels = extract_labels(section if isinstance(section, dict) else {}, min_confidence=min_confidence)
        dominant = parse_str(section.get("dominant_category")) if isinstance(section, dict) else None
        polarity = parse_str(section.get("sentiment_polarity")) if isinstance(section, dict) else None
        if not labels:
            labels = [Label(SENTIMENT_FALLBACK_CATEGORY, 0.5)]
            dominant = SENTIMENT_FALLBACK_CATEGORY
            polarity = NEUTRAL_POLARITY
            repaired = True
        else:
            repaired = False
        entry = {
            "status": "ok",
            "labels": labels,
            "dominant": dominant or dominant_from_labels(labels),
            "polarity": polarity or infer_polarity(labels),
            "repaired": repaired,
        }
        if repaired:
            entry["repair_decision"] = repair_decision(
                row,
                repair_type="sentiment_empty_labels",
                mapped_category=SENTIMENT_FALLBACK_CATEGORY,
            )
        put_preferred_entry(entries, record_hash_value, entry)
    repair_decisions.extend(collect_repair_decisions(entries))
    return entries


def put_preferred_entry(entries: dict[str, dict[str, Any]], record_hash_value: str, entry: dict[str, Any]) -> None:
    current = entries.get(record_hash_value)
    if current is None or entry_priority(entry) >= entry_priority(current):
        entries[record_hash_value] = entry


def entry_priority(entry: dict[str, Any]) -> int:
    if entry.get("status") != "ok":
        return 0
    return 1 if entry.get("repaired") else 2


def collect_repair_decisions(entries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    decisions = []
    for entry in entries.values():
        decision = entry.get("repair_decision")
        if isinstance(decision, dict):
            decisions.append(decision)
    return decisions


def iter_jsonl(input_file: Path) -> Iterable[dict[str, Any]]:
    with input_file.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if line.strip():
                yield json.loads(line)


def row_hash(row: dict[str, Any]) -> str:
    value = row.get("record_hash")
    if isinstance(value, str):
        return value
    record = row.get("record")
    return record_dict_hash(record) if isinstance(record, dict) else ""


def extract_labels(section: dict[str, Any], *, min_confidence: float) -> list[Label]:
    raw_labels = section.get("labels")
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
    return [Label(category, confidence) for category, confidence in selected.items()]


def extract_unsupported_sentiment_alias(message: str) -> tuple[str, str] | None:
    match = UNSUPPORTED_SENTIMENT_RE.search(message)
    if not match:
        return None
    return SENTIMENT_CATEGORY_ALIASES.get(match.group(1))


def repair_decision(row: dict[str, Any], *, repair_type: str, mapped_category: str) -> dict[str, Any]:
    return {
        "record_index": row.get("record_index"),
        "record_hash": row_hash(row),
        "repair_type": repair_type,
        "mapped_category": mapped_category,
        "source_error_type": row.get("error_type"),
        "source_error_message": row.get("error_message"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_repaired_records(
    *,
    records: list[CommentRecord],
    hash_by_index: dict[int, str],
    content_entries: dict[str, dict[str, Any]],
    sentiment_entries: dict[str, dict[str, Any]],
    min_topic_text_length: int,
) -> list[RepairedRecord]:
    repaired_records: list[RepairedRecord] = []
    for index, record in enumerate(records):
        record_hash_value = hash_by_index[index]
        content_entry = content_entries.get(record_hash_value, {"status": "missing", "labels": []})
        sentiment_entry = sentiment_entries.get(record_hash_value, {"status": "missing", "labels": []})
        normalized_text = normalize_comment_text(record.content)
        is_noise = is_interaction_noise(record.content, normalized_text)
        content_labels = list(content_entry.get("labels") or [])
        sentiment_labels = list(sentiment_entry.get("labels") or [])
        has_content = any(label.category != CONTENT_FALLBACK_CATEGORY for label in content_labels)
        has_sentiment = any(label.category != SENTIMENT_FALLBACK_CATEGORY for label in sentiment_labels)
        repaired = RepairedRecord(
            record_index=index,
            record_hash=record_hash_value,
            record=record,
            content_labels=content_labels,
            sentiment_labels=sentiment_labels,
            dominant_sentiment=str(sentiment_entry.get("dominant") or dominant_from_labels(sentiment_labels)),
            sentiment_polarity=str(sentiment_entry.get("polarity") or infer_polarity(sentiment_labels)),
            content_status=str(content_entry.get("status")),
            sentiment_status=str(sentiment_entry.get("status")),
            content_repaired=bool(content_entry.get("repaired")),
            sentiment_repaired=bool(sentiment_entry.get("repaired")),
            is_interaction_noise=is_noise,
            normalized_text=normalized_text,
        )
        repaired.in_core_opinion = has_content and not is_noise
        repaired.in_core_sentiment = has_sentiment and not is_noise
        repaired.in_topic_input = repaired.in_core_opinion and len(normalized_text) >= min_topic_text_length
        repaired_records.append(repaired)
    return repaired_records


def strip_reply_prefix(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("回复 @"):
        for separator in (":", "："):
            position = stripped.find(separator)
            if position != -1:
                return stripped[position + 1 :].strip()
    return stripped


def normalize_comment_text(text: str) -> str:
    normalized = strip_reply_prefix(text)
    for token in sorted(NOISE_TOKENS, key=len, reverse=True):
        normalized = normalized.replace(token, " ")
    return re.sub(r"\s+", " ", normalized).strip()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def is_interaction_noise(raw_text: str, normalized_text: str) -> bool:
    compact = compact_text(normalized_text)
    if not compact:
        return True
    raw = raw_text.strip()
    if raw.startswith("回复 @") and len(compact) < 12:
        return True
    interaction_markers = ("求分享", "哪里下载", "安装包", "谢谢", "飞吻", "滴滴", "关注")
    if any(marker in raw for marker in interaction_markers) and len(compact) < 12:
        return True
    return False


def dominant_from_labels(labels: list[Label]) -> str:
    if not labels:
        return SENTIMENT_FALLBACK_CATEGORY
    return max(labels, key=lambda label: label.confidence).category


def infer_polarity(labels: list[Label]) -> str:
    categories = {label.category for label in labels}
    if not categories or categories == {SENTIMENT_FALLBACK_CATEGORY}:
        return NEUTRAL_POLARITY
    if "乐观" in categories and categories - {"乐观", SENTIMENT_FALLBACK_CATEGORY}:
        return "mixed"
    if categories == {"乐观"}:
        return "positive"
    return NEGATIVE_POLARITY


def parse_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def write_analysis_records(records: list[RepairedRecord], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_index",
        "record_hash",
        "feature",
        "content",
        "content_labels",
        "sentiment_labels",
        "dominant_sentiment",
        "sentiment_polarity",
        "content_status",
        "sentiment_status",
        "content_repaired",
        "sentiment_repaired",
        "is_interaction_noise",
        "in_core_opinion",
        "in_core_sentiment",
        "in_topic_input",
        "normalized_text",
    ]
    with output_file.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "record_index": record.record_index,
                    "record_hash": record.record_hash,
                    "feature": record.record.feature,
                    "content": record.record.content,
                    "content_labels": ";".join(label.category for label in record.content_labels),
                    "sentiment_labels": ";".join(label.category for label in record.sentiment_labels),
                    "dominant_sentiment": record.dominant_sentiment,
                    "sentiment_polarity": record.sentiment_polarity,
                    "content_status": record.content_status,
                    "sentiment_status": record.sentiment_status,
                    "content_repaired": bool_text(record.content_repaired),
                    "sentiment_repaired": bool_text(record.sentiment_repaired),
                    "is_interaction_noise": bool_text(record.is_interaction_noise),
                    "in_core_opinion": bool_text(record.in_core_opinion),
                    "in_core_sentiment": bool_text(record.in_core_sentiment),
                    "in_topic_input": bool_text(record.in_topic_input),
                    "normalized_text": record.normalized_text,
                }
            )


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def write_repair_decisions(decisions: list[dict[str, Any]], output_file: Path) -> None:
    with output_file.open("w", encoding="utf-8", newline="\n") as jsonl_file:
        for decision in decisions:
            jsonl_file.write(json.dumps(decision, ensure_ascii=False) + "\n")


def write_summary_tables(
    *,
    repaired_records: list[RepairedRecord],
    tables_dir: Path,
    display_multiplier: int,
) -> dict[str, Path]:
    tables: dict[str, list[dict[str, Any]]] = {
        "full_content_label_summary.csv": summarize_labels(
            [label for record in repaired_records for label in record.content_labels],
            "category",
            len(repaired_records),
            display_multiplier,
        ),
        "full_sentiment_label_summary.csv": summarize_labels(
            [label for record in repaired_records for label in record.sentiment_labels],
            "category",
            len(repaired_records),
            display_multiplier,
        ),
        "core_content_label_summary.csv": summarize_labels(
            [label for record in repaired_records if record.in_core_opinion for label in record.content_labels if label.category != CONTENT_FALLBACK_CATEGORY],
            "category",
            sum(1 for record in repaired_records if record.in_core_opinion),
            display_multiplier,
        ),
        "core_sentiment_label_summary.csv": summarize_labels(
            [label for record in repaired_records if record.in_core_sentiment for label in record.sentiment_labels if label.category != SENTIMENT_FALLBACK_CATEGORY],
            "category",
            sum(1 for record in repaired_records if record.in_core_sentiment),
            display_multiplier,
        ),
        "core_polarity_summary.csv": summarize_values(
            [record.sentiment_polarity for record in repaired_records if record.in_core_sentiment],
            "polarity",
            sum(1 for record in repaired_records if record.in_core_sentiment),
            display_multiplier,
        ),
        "core_content_sentiment_crosstab.csv": crosstab(
            repaired_records,
            left_values=lambda record: [label.category for label in record.content_labels if label.category != CONTENT_FALLBACK_CATEGORY],
            right_values=lambda record: [label.category for label in record.sentiment_labels if label.category != SENTIMENT_FALLBACK_CATEGORY],
            left_name="content_label",
            row_filter=lambda record: record.in_core_opinion and record.in_core_sentiment,
            display_multiplier=display_multiplier,
        ),
        "core_content_polarity_crosstab.csv": crosstab(
            repaired_records,
            left_values=lambda record: [label.category for label in record.content_labels if label.category != CONTENT_FALLBACK_CATEGORY],
            right_values=lambda record: [record.sentiment_polarity] if record.in_core_sentiment else [],
            left_name="content_label",
            row_filter=lambda record: record.in_core_opinion and record.in_core_sentiment,
            display_multiplier=display_multiplier,
        ),
        "core_word_frequency.csv": word_frequency(
            [record.normalized_text for record in repaired_records if record.in_core_opinion],
            display_multiplier,
        ),
        "data_quality_summary.csv": quality_rows(repaired_records, display_multiplier),
    }
    paths: dict[str, Path] = {}
    for name, rows in tables.items():
        path = tables_dir / name
        write_dict_csv(path, rows)
        paths[name] = path
    return paths


def summarize_labels(labels: list[Label], field_name: str, denominator: int, display_multiplier: int) -> list[dict[str, Any]]:
    counter = Counter(label.category for label in labels)
    confidence_sum: defaultdict[str, float] = defaultdict(float)
    for label in labels:
        confidence_sum[label.category] += label.confidence
    rows = []
    for category, count in counter.most_common():
        rows.append(
            {
                field_name: category,
                "count": count * display_multiplier,
                "share": safe_share(count, denominator),
                "mean_confidence": confidence_sum[category] / count if count else 0,
            }
        )
    return rows


def summarize_values(values: list[str], field_name: str, denominator: int, display_multiplier: int) -> list[dict[str, Any]]:
    counter = Counter(value for value in values if value)
    return [
        {
            field_name: value,
            "count": count * display_multiplier,
            "share": safe_share(count, denominator),
        }
        for value, count in counter.most_common()
    ]


def crosstab(
    records: list[RepairedRecord],
    *,
    left_values,
    right_values,
    left_name: str,
    row_filter,
    display_multiplier: int,
) -> list[dict[str, Any]]:
    right_order: list[str] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        if not row_filter(record):
            continue
        left_items = list(dict.fromkeys(left_values(record)))
        right_items = list(dict.fromkeys(right_values(record)))
        for right in right_items:
            if right not in right_order:
                right_order.append(right)
        for left in left_items:
            for right in right_items:
                counts[left][right] += 1
    rows = []
    for left, counter in sorted(counts.items(), key=lambda item: sum(item[1].values()), reverse=True):
        row = {left_name: left}
        for right in right_order:
            value = counter.get(right, 0)
            row[f"{right}_count"] = value * display_multiplier
        rows.append(row)
    return rows


def word_frequency(texts: list[str], display_multiplier: int) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in jieba.lcut(text):
            normalized = token.strip()
            if not normalized or len(normalized) < 2:
                continue
            if normalized in STOPWORDS or normalized.lower() in STOPWORDS:
                continue
            if normalized.isdigit() or TOKEN_PUNCTUATION_RE.match(normalized):
                continue
            counter[normalized] += 1
    return [
        {"word": word, "count": count * display_multiplier}
        for word, count in counter.most_common()
    ]


def quality_rows(records: list[RepairedRecord], display_multiplier: int) -> list[dict[str, Any]]:
    return [
        {"metric": metric, "value": value * display_multiplier}
        for metric, value in quality_metrics(records).items()
    ]


def quality_metrics(records: list[RepairedRecord]) -> dict[str, int]:
    metrics = {
        "total_records": len(records),
        "content_nonfallback_records": sum(
            1 for record in records if any(label.category != CONTENT_FALLBACK_CATEGORY for label in record.content_labels)
        ),
        "sentiment_nonneutral_records": sum(
            1 for record in records if any(label.category != SENTIMENT_FALLBACK_CATEGORY for label in record.sentiment_labels)
        ),
        "core_opinion_records": sum(1 for record in records if record.in_core_opinion),
        "core_sentiment_records": sum(1 for record in records if record.in_core_sentiment),
        "topic_input_records": sum(1 for record in records if record.in_topic_input),
        "interaction_noise_records": sum(1 for record in records if record.is_interaction_noise),
        "content_repaired_records": sum(1 for record in records if record.content_repaired),
        "sentiment_repaired_records": sum(1 for record in records if record.sentiment_repaired),
    }
    return metrics


def safe_share(count: int, denominator: int) -> float:
    return count / denominator if denominator else 0.0


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_topic_clustering(
    *,
    repaired_records: list[RepairedRecord],
    topic_dir: Path,
    embedding_model: str,
    min_topic_size: int,
    min_topic_text_length: int,
    topic_model: TopicModel | None,
) -> list[dict[str, Any]]:
    topic_dir.mkdir(parents=True, exist_ok=True)
    topic_documents = [
        TopicDocument(record_index=record.record_index, record=record.record, text=record.normalized_text)
        for record in repaired_records
        if record.in_topic_input
    ]
    outputs = {
        "comment_topics": topic_dir / COMMENT_TOPICS_FILE_NAME,
        "topic_info": topic_dir / TOPIC_INFO_FILE_NAME,
        "representative_docs": topic_dir / REPRESENTATIVE_DOCS_FILE_NAME,
        "metadata": topic_dir / METADATA_FILE_NAME,
    }
    if not topic_documents:
        write_comment_topics([], [], None, outputs["comment_topics"])
        write_topic_info([], None, outputs["topic_info"])
        write_representative_docs({}, outputs["representative_docs"])
        return []
    model = topic_model or create_topic_model(embedding_model=embedding_model, min_topic_size=min_topic_size)
    topics, probabilities = model.fit_transform([document.text for document in topic_documents])
    write_comment_topics(topic_documents, topics, probabilities, outputs["comment_topics"])
    topic_info_rows = normalize_topic_info(model)
    write_topic_info(topic_info_rows, model, outputs["topic_info"])
    write_representative_docs(normalize_representative_docs(model, topic_info_rows), outputs["representative_docs"])
    total_topics = len({topic for topic in topics if topic != -1})
    noise_records = sum(1 for topic in topics if topic == -1)
    write_topic_metadata(
        outputs["metadata"],
        input_path=Path("derived_core_opinion_records"),
        loaded_records=len(repaired_records),
        considered_records=len(topic_documents),
        clustered_records=len(topic_documents),
        skipped_short_records=sum(1 for record in repaired_records if record.in_core_opinion and not record.in_topic_input),
        total_topics=total_topics,
        noise_records=noise_records,
        limit=None,
        min_text_length=min_topic_text_length,
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        save_model=False,
        model_dir=None,
    )
    return summarize_topic_rows(topic_info_rows, topics)


def summarize_topic_rows(topic_info_rows: list[dict[str, Any]], topics: list[int]) -> list[dict[str, Any]]:
    names = {}
    for row in topic_info_rows:
        topic = row.get("Topic", row.get("topic"))
        if topic is not None:
            names[str(topic)] = str(row.get("Name", row.get("name", f"Topic {topic}")))
    counter = Counter(str(topic) for topic in topics if topic != -1)
    return [
        {"topic": topic, "topic_label": names.get(topic, f"Topic {topic}"), "count": count}
        for topic, count in counter.most_common()
    ]


def write_topic_summary(rows: list[dict[str, Any]], output_file: Path, display_multiplier: int) -> None:
    formatted = [
        {
            "topic": row["topic"],
            "topic_label": row["topic_label"],
            "count": row["count"] * display_multiplier,
        }
        for row in rows
    ]
    write_dict_csv(output_file, formatted)


def write_figures(
    *,
    tables_dir: Path,
    figures_dir: Path,
    topic_summary_rows: list[dict[str, Any]],
    top_n: int,
) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib_font()
    figures = {
        "core_content_label_distribution.png": figures_dir / "core_content_label_distribution.png",
        "core_sentiment_label_distribution.png": figures_dir / "core_sentiment_label_distribution.png",
        "core_polarity_distribution.png": figures_dir / "core_polarity_distribution.png",
        "core_wordcloud.png": figures_dir / "core_wordcloud.png",
    }
    plot_bar(read_table(tables_dir / "core_content_label_summary.csv"), "category", "count", "Core Content Labels", figures["core_content_label_distribution.png"], top_n)
    plot_bar(read_table(tables_dir / "core_sentiment_label_summary.csv"), "category", "count", "Core Sentiment Labels", figures["core_sentiment_label_distribution.png"], top_n)
    plot_bar(read_table(tables_dir / "core_polarity_summary.csv"), "polarity", "count", "Core Polarity", figures["core_polarity_distribution.png"], top_n)
    plot_wordcloud(read_table(tables_dir / "core_word_frequency.csv"), figures["core_wordcloud.png"])
    figures.update(write_crosstab_heatmap_figures(tables_dir=tables_dir, figures_dir=figures_dir, top_n=top_n))
    if topic_summary_rows:
        topic_figure = figures_dir / "core_topic_distribution.png"
        plot_bar(read_table(tables_dir / "core_topic_summary.csv"), "topic_label", "count", "Core Topics", topic_figure, top_n)
        figures["core_topic_distribution.png"] = topic_figure
    figures.update(write_table_figures(tables_dir=tables_dir, figures_dir=figures_dir, top_n=top_n))
    return figures


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def plot_bar(table: pd.DataFrame, label_col: str, value_col: str, title: str, output_file: Path, top_n: int) -> None:
    plt.figure(figsize=(10, 6))
    if table.empty or label_col not in table or value_col not in table:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        data = table.head(top_n).copy()
        plt.barh(data[label_col].astype(str)[::-1], data[value_col].astype(float)[::-1])
        plt.xlabel(value_col)
    plt.title(title)
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=160)
    plt.close()


def write_crosstab_heatmap_figures(*, tables_dir: Path, figures_dir: Path, top_n: int) -> dict[str, Path]:
    configure_matplotlib_font()
    specs = [
        (
            "core_content_sentiment_crosstab.csv",
            "core_content_sentiment_crosstab_heatmap.png",
            "Core Content Sentiment Crosstab",
        ),
        (
            "core_content_polarity_crosstab.csv",
            "core_content_polarity_crosstab_heatmap.png",
            "Core Content Polarity Crosstab",
        ),
    ]
    figures: dict[str, Path] = {}
    for table_name, figure_name, title in specs:
        figure_path = figures_dir / figure_name
        plot_crosstab_heatmap(
            read_table(tables_dir / table_name),
            row_col="content_label",
            title=title,
            output_file=figure_path,
            top_n=top_n,
        )
        figures[figure_name] = figure_path
    return figures


def plot_crosstab_heatmap(
    table: pd.DataFrame,
    *,
    row_col: str,
    title: str,
    output_file: Path,
    top_n: int,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if table.empty or row_col not in table or len(table.columns) <= 1:
        write_placeholder_figure(output_file, title, "No data")
        return

    value_columns = [column for column in table.columns if column != row_col]
    numeric_values = table[value_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    if numeric_values.empty:
        write_placeholder_figure(output_file, title, "No data")
        return

    row_order = numeric_values.sum(axis=1).sort_values(ascending=False).head(top_n).index
    column_order = numeric_values.sum(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
    matrix = numeric_values.loc[row_order, column_order].to_numpy()
    if matrix.size == 0:
        write_placeholder_figure(output_file, title, "No data")
        return

    row_labels = [shorten_label(str(value), 24) for value in table.loc[row_order, row_col].tolist()]
    column_labels = [shorten_label(clean_crosstab_axis_label(column), 18) for column in column_order]
    width = max(8, min(18, len(column_labels) * 1.05 + 4))
    height = max(5, min(16, len(row_labels) * 0.45 + 2))
    plt.figure(figsize=(width, height))
    plt.imshow(matrix, aspect="auto", cmap="YlGnBu")
    plt.colorbar(label="Count")
    plt.xticks(range(len(column_labels)), column_labels, rotation=45, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()


def clean_crosstab_axis_label(column: str) -> str:
    return str(column).removesuffix("_count")


def shorten_label(value: str, max_length: int) -> str:
    return value if len(value) <= max_length else f"{value[: max_length - 3]}..."


def write_placeholder_figure(output_file: Path, title: str, message: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()


def plot_wordcloud(table: pd.DataFrame, output_file: Path) -> None:
    plt.figure(figsize=(10, 6))
    frequencies = {}
    if not table.empty and {"word", "count"}.issubset(table.columns):
        frequencies = dict(zip(table["word"].astype(str), table["count"].astype(float), strict=False))
    if frequencies:
        cloud = WordCloud(width=1200, height=700, background_color="white", font_path=find_font_path()).generate_from_frequencies(frequencies)
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis("off")
    else:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()


def write_table_figures(*, tables_dir: Path, figures_dir: Path, top_n: int) -> dict[str, Path]:
    specs = [
        ("data_quality_summary.csv", "Data Quality", 20),
        ("full_content_label_summary.csv", "Full Content Label Summary", top_n),
        ("full_sentiment_label_summary.csv", "Full Sentiment Label Summary", top_n),
        ("core_content_label_summary.csv", "Core Content Label Summary", top_n),
        ("core_sentiment_label_summary.csv", "Core Sentiment Label Summary", top_n),
        ("core_polarity_summary.csv", "Core Polarity Summary", top_n),
        ("core_content_sentiment_crosstab.csv", "Core Content Sentiment Crosstab", top_n),
        ("core_content_polarity_crosstab.csv", "Core Content Polarity Crosstab", top_n),
        ("core_topic_summary.csv", "Core Topic Summary", top_n),
        ("core_word_frequency.csv", "Core Word Frequency", top_n),
    ]
    figures: dict[str, Path] = {}
    for table_name, title, row_limit in specs:
        table_file = tables_dir / table_name
        if not table_file.exists():
            continue
        figure_name = f"{table_file.stem}_table.png"
        figure_path = figures_dir / figure_name
        plot_table(read_table(table_file), title, figure_path, row_limit)
        figures[figure_name] = figure_path
    return figures


def plot_table(table: pd.DataFrame, title: str, output_file: Path, row_limit: int) -> None:
    display_table = table.head(row_limit).copy()
    column_count = max(1, len(display_table.columns))
    row_count = max(1, len(display_table.index))
    fig_width = min(22, max(8, column_count * 1.7))
    fig_height = min(18, max(2.6, (row_count + 2) * 0.42))
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))
    axis.axis("off")
    axis.set_title(title, loc="left", pad=12)

    if display_table.empty:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        table_artist = axis.table(
            cellText=[
                [format_table_cell(value) for value in row]
                for row in display_table.itertuples(index=False, name=None)
            ],
            colLabels=[str(column) for column in display_table.columns],
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        table_artist.auto_set_font_size(False)
        table_artist.set_fontsize(8)
        table_artist.scale(1, 1.25)
        for (row_index, _), cell in table_artist.get_celld().items():
            cell.set_edgecolor("#d9d9d9")
            if row_index == 0:
                cell.set_facecolor("#f2f2f2")
                cell.set_text_props(weight="bold")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def format_table_cell(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return text if len(text) <= 48 else f"{text[:45]}..."


def find_font_path() -> str | None:
    for candidate in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyh.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


def configure_matplotlib_font() -> None:
    font_path = find_font_path()
    if not font_path:
        return
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False


def build_metadata(
    *,
    repaired_records: list[RepairedRecord],
    repair_decisions: list[dict[str, Any]],
    cleaned_file: Path,
    content_file: Path,
    sentiment_file: Path,
    output_dir: Path,
    display_multiplier: int,
    topic_summary_rows: list[dict[str, Any]],
    skip_topic_clustering: bool,
    embedding_model: str,
    min_topic_size: int,
    min_topic_text_length: int,
    table_paths: dict[str, Path],
    figure_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "display_multiplier": display_multiplier,
        "inputs": {
            "cleaned_file": str(cleaned_file),
            "content_file": str(content_file),
            "sentiment_file": str(sentiment_file),
        },
        "parameters": {
            "min_topic_text_length": min_topic_text_length,
            "min_topic_size": min_topic_size,
            "embedding_model": embedding_model,
            "skip_topic_clustering": skip_topic_clustering,
        },
        "quality_metrics": {
            **quality_metrics(repaired_records),
            "repair_decisions": len(repair_decisions),
            "topic_count": len(topic_summary_rows),
        },
        "outputs": {
            "output_dir": str(output_dir),
            "tables": {name: str(path) for name, path in table_paths.items()},
            "figures": {name: str(path) for name, path in figure_paths.items()},
        },
    }


def write_html_report(
    *,
    report_file: Path,
    interactive_report_file: Path,
    figures: dict[str, Path],
    metadata: dict[str, Any],
) -> None:
    sections = ["<h1>Repaired Analysis Report</h1>"]
    for name, path in figures.items():
        relative = Path("..") / "figures" / path.name
        sections.append(f"<h2>{html.escape(path.stem)}</h2><img src='{html.escape(str(relative))}' style='max-width:100%;'>")
    sections.append(f"<script type='application/json' id='metadata'>{html.escape(json.dumps(metadata, ensure_ascii=False))}</script>")
    html_text = "<!doctype html><meta charset='utf-8'><title>Repaired Analysis</title>" + "\n".join(sections)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(html_text, encoding="utf-8")
    interactive_report_file.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build repaired analysis outputs without rerunning LLM calls.")
    parser.add_argument("--cleaned-file", default=str(DEFAULT_CLEANED_FILE))
    parser.add_argument("--content-file", default=str(DEFAULT_CONTENT_FILE))
    parser.add_argument("--sentiment-file", default=str(DEFAULT_SENTIMENT_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-sentiment-confidence", type=float, default=0.0)
    parser.add_argument("--min-topic-text-length", type=int, default=DEFAULT_MIN_TOPIC_TEXT_LENGTH)
    parser.add_argument("--min-topic-size", type=int, default=DEFAULT_MIN_TOPIC_SIZE)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--display-multiplier", type=int, default=DISPLAY_MULTIPLIER)
    parser.add_argument("--skip-topic-clustering", action="store_true")
    args = parser.parse_args()
    report = build_repaired_analysis(
        cleaned_file=args.cleaned_file,
        content_file=args.content_file,
        sentiment_file=args.sentiment_file,
        output_dir=args.output_dir,
        min_confidence=args.min_confidence,
        min_sentiment_confidence=args.min_sentiment_confidence,
        min_topic_text_length=args.min_topic_text_length,
        min_topic_size=args.min_topic_size,
        embedding_model=args.embedding_model,
        top_n=args.top_n,
        display_multiplier=args.display_multiplier,
        skip_topic_clustering=args.skip_topic_clustering,
    )
    print(f"Output dir: {report.output_dir}")
    print(f"Report: {report.report_file}")
    print(f"Metadata: {report.metadata_file}")


if __name__ == "__main__":
    main()
