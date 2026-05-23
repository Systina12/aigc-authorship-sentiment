import csv
import json

import pytest

from scripts.build_analysis_report import build_analysis_report


def test_build_analysis_report_writes_tables_figures_and_html(tmp_path):
    cleaned_file = tmp_path / "comments_cleaned.csv"
    content_file = tmp_path / "comment_labels.jsonl"
    sentiment_file = tmp_path / "comment_sentiment.jsonl"
    topic_file = tmp_path / "comment_topics.csv"
    topic_info_file = tmp_path / "topic_info.csv"
    output_dir = tmp_path / "analysis_report"
    write_comment_csv(
        cleaned_file,
        [
            "AI 工具 提高 效率",
            "版权 争议 让 创作者 焦虑",
            "普通 无关 评论",
            "AI 工具 学习",
        ],
    )
    write_jsonl(
        content_file,
        [
            content_row(0, "hash-0", [("技术认可", 0.9), ("工具化认知", 0.85)]),
            content_row(1, "hash-1", [("职业焦虑", 0.95), ("版权争议", 0.5)]),
            content_row(2, "hash-2", [("无法归类/无关讨论", 1.0)]),
            {"record_index": 3, "record_hash": "hash-3", "status": "error", "analysis": None},
        ],
    )
    write_jsonl(
        sentiment_file,
        [
            sentiment_row(0, "hash-0", [("乐观", 0.9), ("质疑", 0.4)], dominant="乐观", polarity="positive"),
            sentiment_row(1, "hash-1", [("焦虑", 0.9)], dominant="焦虑", polarity="negative"),
            sentiment_row(2, "hash-2", [("中性/无法判断", 1.0)], dominant="中性/无法判断", polarity="neutral"),
            {"record_index": 3, "record_hash": "hash-3", "status": "error", "sentiment": None},
        ],
    )
    write_topic_csv(
        topic_file,
        [
            {"record_index": "0", "topic": "0", "topic_probability": "0.9"},
            {"record_index": "1", "topic": "1", "topic_probability": "0.8"},
            {"record_index": "2", "topic": "-1", "topic_probability": "0.2"},
        ],
    )
    write_topic_info_csv(topic_info_file)

    report = build_analysis_report(
        cleaned_file=cleaned_file,
        content_file=content_file,
        sentiment_file=sentiment_file,
        topic_file=topic_file,
        topic_info_file=topic_info_file,
        cooccurrence_dir=tmp_path / "missing_cooccurrence",
        output_dir=output_dir,
        min_confidence=0.8,
        min_sentiment_confidence=0.6,
        top_n=5,
    )

    content_summary = read_csv(output_dir / "content_label_summary.csv")
    sentiment_summary = read_csv(output_dir / "sentiment_label_summary.csv")
    polarity_summary = read_csv(output_dir / "sentiment_polarity_summary.csv")
    dominant_summary = read_csv(output_dir / "dominant_sentiment_summary.csv")
    content_sentiment = read_csv(output_dir / "content_sentiment_crosstab.csv")
    content_polarity = read_csv(output_dir / "content_polarity_crosstab.csv")
    topic_summary = read_csv(output_dir / "topic_summary.csv")
    topic_content = read_csv(output_dir / "topic_content_crosstab.csv")
    topic_sentiment = read_csv(output_dir / "topic_sentiment_crosstab.csv")
    word_frequency = read_csv(output_dir / "word_frequency.csv")
    quality = read_csv(output_dir / "data_quality_summary.csv")
    metadata = json.loads((output_dir / "report_metadata.json").read_text(encoding="utf-8"))

    assert report.output_dir == output_dir
    assert category_counts(content_summary) == {
        "技术认可": "1",
        "工具化认知": "1",
        "职业焦虑": "1",
        "无法归类/无关讨论": "1",
    }
    assert "版权争议" not in category_counts(content_summary)
    assert category_counts(sentiment_summary) == {"乐观": "1", "焦虑": "1", "中性/无法判断": "1"}
    assert count_by_field(polarity_summary, "polarity") == {"positive": "1", "negative": "1", "neutral": "1"}
    assert count_by_field(dominant_summary, "category") == {"乐观": "1", "焦虑": "1", "中性/无法判断": "1"}
    assert row_value(content_sentiment, "content_label", "技术认可", "乐观") == "1"
    assert row_value(content_sentiment, "content_label", "职业焦虑", "焦虑") == "1"
    assert "质疑" not in content_sentiment[0]
    assert row_value(content_polarity, "content_label", "工具化认知", "positive") == "1"
    assert row_value(content_polarity, "content_label", "职业焦虑", "negative") == "1"
    assert count_by_field(topic_summary, "topic") == {"0": "1", "1": "1"}
    assert "-1" not in count_by_field(topic_summary, "topic")
    assert row_value(topic_content, "topic", "0", "技术认可") == "1"
    assert row_value(topic_content, "topic", "1", "职业焦虑") == "1"
    assert row_value(topic_sentiment, "topic", "0", "乐观") == "1"
    assert row_value(topic_sentiment, "topic", "1", "焦虑") == "1"
    assert int(word_frequency[0]["count"]) >= 1
    assert quality_value(quality, "content_error_rows") == "1"
    assert quality_value(quality, "sentiment_error_rows") == "1"
    assert quality_value(quality, "topic_noise_rows") == "1"
    assert metadata["inputs"]["topic_file"]["status"] == "loaded"
    assert metadata["quality_metrics"]["matched_content_sentiment_records"] == 3
    assert metadata["quality_metrics"]["topic_noise_rows"] == 1

    expected_figures = [
        "content_label_distribution.png",
        "sentiment_label_distribution.png",
        "sentiment_polarity_distribution.png",
        "content_sentiment_heatmap.png",
        "topic_distribution.png",
        "wordcloud.png",
        "content_sentiment_network.png",
    ]
    for figure_name in expected_figures:
        figure_file = output_dir / "figures" / figure_name
        assert figure_file.exists()
        assert figure_file.stat().st_size > 0

    assert (output_dir / "report.html").stat().st_size > 0
    assert (output_dir / "interactive_report.html").stat().st_size > 0


def test_build_analysis_report_skips_missing_optional_inputs(tmp_path):
    cleaned_file = tmp_path / "comments_cleaned.csv"
    content_file = tmp_path / "comment_labels.jsonl"
    sentiment_file = tmp_path / "comment_sentiment.jsonl"
    output_dir = tmp_path / "analysis_report"
    write_comment_csv(cleaned_file, ["AI 工具 提高 效率"])
    write_jsonl(content_file, [content_row(0, "hash-0", [("技术认可", 0.9)])])
    write_jsonl(sentiment_file, [sentiment_row(0, "hash-0", [("乐观", 0.9)], dominant="乐观", polarity="positive")])

    build_analysis_report(
        cleaned_file=cleaned_file,
        content_file=content_file,
        sentiment_file=sentiment_file,
        topic_file=tmp_path / "missing_topics.csv",
        topic_info_file=tmp_path / "missing_topic_info.csv",
        cooccurrence_dir=tmp_path / "missing_cooccurrence",
        output_dir=output_dir,
    )

    metadata = json.loads((output_dir / "report_metadata.json").read_text(encoding="utf-8"))

    assert metadata["inputs"]["topic_file"]["status"] == "missing"
    assert metadata["inputs"]["cooccurrence_dir"]["status"] == "missing"
    assert not (output_dir / "topic_summary.csv").exists()
    assert (output_dir / "content_label_summary.csv").exists()
    assert (output_dir / "report.html").exists()


def test_build_analysis_report_removes_stale_topic_tables_when_topic_file_is_missing(tmp_path):
    cleaned_file = tmp_path / "comments_cleaned.csv"
    content_file = tmp_path / "comment_labels.jsonl"
    sentiment_file = tmp_path / "comment_sentiment.jsonl"
    topic_file = tmp_path / "comment_topics.csv"
    topic_info_file = tmp_path / "topic_info.csv"
    output_dir = tmp_path / "analysis_report"
    write_comment_csv(cleaned_file, ["alpha beta gamma"])
    write_jsonl(content_file, [content_row(0, "hash-0", [("content-a", 0.9)])])
    write_jsonl(sentiment_file, [sentiment_row(0, "hash-0", [("sentiment-a", 0.9)], dominant="sentiment-a", polarity="positive")])
    write_topic_csv(topic_file, [{"record_index": "0", "topic": "0", "topic_probability": "0.9"}])
    write_topic_info_csv(topic_info_file)

    build_analysis_report(
        cleaned_file=cleaned_file,
        content_file=content_file,
        sentiment_file=sentiment_file,
        topic_file=topic_file,
        topic_info_file=topic_info_file,
        output_dir=output_dir,
    )

    assert (output_dir / "topic_summary.csv").exists()
    assert (output_dir / "topic_content_crosstab.csv").exists()
    assert (output_dir / "topic_sentiment_crosstab.csv").exists()

    build_analysis_report(
        cleaned_file=cleaned_file,
        content_file=content_file,
        sentiment_file=sentiment_file,
        topic_file=tmp_path / "missing_topics.csv",
        topic_info_file=topic_info_file,
        output_dir=output_dir,
    )

    metadata = json.loads((output_dir / "report_metadata.json").read_text(encoding="utf-8"))

    assert metadata["inputs"]["topic_file"]["status"] == "missing"
    assert not (output_dir / "topic_summary.csv").exists()
    assert not (output_dir / "topic_content_crosstab.csv").exists()
    assert not (output_dir / "topic_sentiment_crosstab.csv").exists()


def test_build_analysis_report_keeps_running_when_topic_info_is_bad(tmp_path):
    cleaned_file = tmp_path / "comments_cleaned.csv"
    content_file = tmp_path / "comment_labels.jsonl"
    sentiment_file = tmp_path / "comment_sentiment.jsonl"
    topic_file = tmp_path / "comment_topics.csv"
    topic_info_file = tmp_path / "bad_topic_info.csv"
    output_dir = tmp_path / "analysis_report"
    write_comment_csv(cleaned_file, ["alpha beta gamma"])
    write_jsonl(content_file, [content_row(0, "hash-0", [("content-a", 0.9)])])
    write_jsonl(sentiment_file, [sentiment_row(0, "hash-0", [("sentiment-a", 0.9)], dominant="sentiment-a", polarity="positive")])
    write_topic_csv(topic_file, [{"record_index": "0", "topic": "0", "topic_probability": "0.9"}])
    topic_info_file.write_bytes(b"\xff\xff\xff")

    build_analysis_report(
        cleaned_file=cleaned_file,
        content_file=content_file,
        sentiment_file=sentiment_file,
        topic_file=topic_file,
        topic_info_file=topic_info_file,
        output_dir=output_dir,
    )

    metadata = json.loads((output_dir / "report_metadata.json").read_text(encoding="utf-8"))
    topic_summary = read_csv(output_dir / "topic_summary.csv")

    assert metadata["inputs"]["topic_file"]["status"] == "loaded"
    assert metadata["inputs"]["topic_info_file"]["status"] == "error"
    assert metadata["warnings"]
    assert topic_summary[0]["topic"] == "0"
    assert topic_summary[0]["topic_label"] == "Topic 0"


def test_build_analysis_report_validates_parameters(tmp_path):
    with pytest.raises(ValueError, match="limit"):
        build_analysis_report(output_dir=tmp_path / "out", limit=-1)

    with pytest.raises(ValueError, match="top_n"):
        build_analysis_report(output_dir=tmp_path / "out", top_n=0)

    with pytest.raises(ValueError, match="min_confidence"):
        build_analysis_report(output_dir=tmp_path / "out", min_confidence=2)

    with pytest.raises(ValueError, match="min_sentiment_confidence"):
        build_analysis_report(output_dir=tmp_path / "out", min_sentiment_confidence=-0.1)


def write_comment_csv(path, contents):
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["username", "gender", "content", "comment_time", "likes", "ip_location", "signature", "feature"],
        )
        writer.writeheader()
        for index, content in enumerate(contents):
            writer.writerow(
                {
                    "username": f"user-{index}",
                    "gender": "",
                    "content": content,
                    "comment_time": "2026-05-10 10:00:00",
                    "likes": "1",
                    "ip_location": "",
                    "signature": "",
                    "feature": "aigc",
                }
            )


def content_row(record_index, record_hash, labels):
    return {
        "record_index": record_index,
        "record_hash": record_hash,
        "status": "ok",
        "analysis": {
            "summary": "test",
            "labels": normalize_labels(labels),
        },
    }


def sentiment_row(record_index, record_hash, labels, *, dominant, polarity):
    return {
        "record_index": record_index,
        "record_hash": record_hash,
        "status": "ok",
        "sentiment": {
            "summary": "test",
            "dominant_category": dominant,
            "sentiment_polarity": polarity,
            "labels": normalize_labels(labels),
        },
    }


def normalize_labels(labels):
    normalized_labels = []
    for label in labels:
        if isinstance(label, tuple):
            category, confidence = label
        else:
            category, confidence = label, 1.0
        normalized_labels.append({"category": category, "confidence": confidence, "rationale": "test"})
    return normalized_labels


def write_topic_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["record_index", "topic", "topic_probability"])
        writer.writeheader()
        writer.writerows(rows)


def write_topic_info_csv(path):
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["Topic", "Name", "Keywords"])
        writer.writeheader()
        writer.writerow({"Topic": "0", "Name": "ai_tools", "Keywords": "AI:0.2;工具:0.1"})
        writer.writerow({"Topic": "1", "Name": "copyright", "Keywords": "版权:0.3;训练:0.2"})


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def category_counts(rows):
    return count_by_field(rows, "category")


def count_by_field(rows, field):
    return {row[field]: row["count"] for row in rows}


def row_value(rows, row_field, row_name, column_name):
    for row in rows:
        if row[row_field] == row_name:
            return row[column_name]
    raise AssertionError(f"Missing row {row_name!r}")


def quality_value(rows, metric):
    for row in rows:
        if row["metric"] == metric:
            return row["value"]
    raise AssertionError(f"Missing metric {metric!r}")
