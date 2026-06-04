import csv
import json
from pathlib import Path

from scripts.analyze_content import record_dict_hash
from scripts.build_repaired_analysis import build_repaired_analysis


LAST_CLEANED_HASHES_BY_INDEX = {}


class FakeTopicModel:
    def __init__(self):
        self.fit_documents = None

    def fit_transform(self, documents):
        self.fit_documents = documents
        return [0 for _ in documents], [0.9 for _ in documents]

    def get_topic_info(self):
        return [{"Topic": 0, "Count": len(self.fit_documents or []), "Name": "0_core_ai"}]

    def get_topic(self, topic):
        return [("AI", 0.5), ("创作", 0.3)]

    def get_representative_docs(self, topic=None):
        return {0: self.fit_documents or []} if topic is None else (self.fit_documents or [])

    def save(self, path):
        return None


def test_build_repaired_analysis_repairs_core_outputs_without_raw_or_x10_labels(tmp_path):
    cleaned_file = tmp_path / "comments_cleaned.csv"
    content_file = tmp_path / "comment_labels.jsonl"
    sentiment_file = tmp_path / "comment_sentiment.jsonl"
    output_dir = tmp_path / "repaired_analysis"
    write_comment_csv(
        cleaned_file,
        [
            "AI 工具 提高 效率",
            "版权 争议 让 创作者 失望",
            "普通 无关 评论",
            "知亿点人工智能适合国内用玫瑰公主浩害羞",
            "回复 @用户 : 求分享 谢谢 飞吻",
        ],
    )
    write_jsonl(
        content_file,
        [
            content_row(0, [("技术认可", 0.9), ("工具化认知", 0.8)]),
            content_row(1, [("版权争议", 0.85)]),
            content_row(2, [("无法归类/无关讨论", 1.0)]),
            content_row(3, []),
            content_row(4, [("技术认可", 0.7)]),
        ],
    )
    write_jsonl(
        sentiment_file,
        [
            sentiment_row(0, [("乐观", 0.9)], dominant="乐观", polarity="positive"),
            {
                "record_index": 1,
                "record_hash": LAST_CLEANED_HASHES_BY_INDEX[1],
                "record": record_dict("版权 争议 让 创作者 失望", 1),
                "status": "error",
                "sentiment": None,
                "error_type": "ValueError",
                "error_message": "LLM returned unsupported sentiment category: '失望'",
            },
            sentiment_row(2, [("中性/无法判断", 1.0)], dominant="中性/无法判断", polarity="neutral"),
            sentiment_row(3, [("中性/无法判断", 0.8)], dominant="中性/无法判断", polarity="neutral"),
            sentiment_row(4, [("乐观", 0.8)], dominant="乐观", polarity="positive"),
        ],
    )
    topic_model = FakeTopicModel()

    report = build_repaired_analysis(
        cleaned_file=cleaned_file,
        content_file=content_file,
        sentiment_file=sentiment_file,
        output_dir=output_dir,
        min_topic_size=2,
        topic_model=topic_model,
    )

    analysis_rows = read_csv(output_dir / "tables" / "analysis_records.csv")
    content_summary = read_csv(output_dir / "tables" / "core_content_label_summary.csv")
    sentiment_summary = read_csv(output_dir / "tables" / "core_sentiment_label_summary.csv")
    polarity_summary = read_csv(output_dir / "tables" / "core_polarity_summary.csv")
    quality = read_csv(output_dir / "tables" / "data_quality_summary.csv")
    decisions = read_jsonl(output_dir / "repair_decisions.jsonl")
    metadata = json.loads((output_dir / "repair_metadata.json").read_text(encoding="utf-8"))

    assert report.output_dir == output_dir
    assert len(analysis_rows) == 5
    assert analysis_rows[3]["content_labels"] == "无法归类/无关讨论"
    assert analysis_rows[3]["content_repaired"] == "true"
    assert analysis_rows[1]["sentiment_labels"] == "焦虑"
    assert analysis_rows[1]["sentiment_repaired"] == "true"
    assert analysis_rows[4]["is_interaction_noise"] == "true"
    assert analysis_rows[4]["in_core_opinion"] == "false"

    assert count_by_category(content_summary) == {
        "技术认可": "10",
        "工具化认知": "10",
        "版权争议": "10",
    }
    assert count_by_category(sentiment_summary) == {
        "乐观": "10",
        "焦虑": "10",
    }
    assert count_by_field(polarity_summary, "polarity") == {"positive": "10", "negative": "10"}
    assert quality_value(quality, "total_records") == "50"
    assert quality_value(quality, "core_opinion_records") == "20"
    assert quality_value(quality, "interaction_noise_records") == "10"
    assert_table_headers_are_clean(content_summary)
    assert_table_headers_are_clean(sentiment_summary)
    assert_table_headers_are_clean(polarity_summary)
    assert_table_headers_are_clean(quality)
    assert {decision["repair_type"] for decision in decisions} == {"content_empty_labels", "sentiment_category_alias"}
    assert topic_model.fit_documents == ["AI 工具 提高 效率", "版权 争议 让 创作者 失望"]
    assert (output_dir / "topic_clustering" / "comment_topics.csv").exists()
    assert (output_dir / "report" / "report.html").exists()
    report_html = (output_dir / "report" / "report.html").read_text(encoding="utf-8")
    assert "<table" not in report_html.lower()
    assert_clean_text(report_html)
    figure_names = {path.name for path in (output_dir / "figures").glob("*.png")}
    assert {
        "core_content_label_distribution.png",
        "core_sentiment_label_distribution.png",
        "core_polarity_distribution.png",
        "core_wordcloud.png",
        "core_topic_distribution.png",
        "data_quality_summary_table.png",
        "core_content_label_summary_table.png",
        "core_sentiment_label_summary_table.png",
        "core_polarity_summary_table.png",
        "core_topic_summary_table.png",
        "core_word_frequency_table.png",
    }.issubset(figure_names)
    for name in figure_names:
        assert_clean_text(name)
    assert metadata["quality_metrics"]["core_opinion_records"] == 2


def test_build_repaired_analysis_prefers_ok_entries_over_later_errors(tmp_path):
    cleaned_file = tmp_path / "comments_cleaned.csv"
    content_file = tmp_path / "comment_labels.jsonl"
    sentiment_file = tmp_path / "comment_sentiment.jsonl"
    output_dir = tmp_path / "repaired_analysis"
    write_comment_csv(cleaned_file, ["AI 工具 提高 效率"])
    write_jsonl(
        content_file,
        [
            content_row(0, [("技术认可", 0.9)]),
            {
                "record_index": 0,
                "record_hash": LAST_CLEANED_HASHES_BY_INDEX[0],
                "record": record_dict("AI 工具 提高 效率", 0),
                "status": "error",
                "analysis": None,
                "error_type": "HTTPError",
                "error_message": "temporary gateway error",
            },
        ],
    )
    write_jsonl(
        sentiment_file,
        [
            sentiment_row(0, [("乐观", 0.9)], dominant="乐观", polarity="positive"),
            {
                "record_index": 0,
                "record_hash": LAST_CLEANED_HASHES_BY_INDEX[0],
                "record": record_dict("AI 工具 提高 效率", 0),
                "status": "error",
                "sentiment": None,
                "error_type": "JSONDecodeError",
                "error_message": "Expecting value: line 1 column 1 (char 0)",
            },
        ],
    )

    build_repaired_analysis(
        cleaned_file=cleaned_file,
        content_file=content_file,
        sentiment_file=sentiment_file,
        output_dir=output_dir,
        skip_topic_clustering=True,
    )

    analysis_rows = read_csv(output_dir / "tables" / "analysis_records.csv")
    assert analysis_rows[0]["content_status"] == "ok"
    assert analysis_rows[0]["sentiment_status"] == "ok"
    assert analysis_rows[0]["content_labels"] == "技术认可"
    assert analysis_rows[0]["sentiment_labels"] == "乐观"
    assert analysis_rows[0]["in_core_opinion"] == "true"
    assert analysis_rows[0]["in_core_sentiment"] == "true"


def test_build_repaired_analysis_removes_stale_topic_outputs_when_topic_is_skipped(tmp_path):
    cleaned_file = tmp_path / "comments_cleaned.csv"
    content_file = tmp_path / "comment_labels.jsonl"
    sentiment_file = tmp_path / "comment_sentiment.jsonl"
    output_dir = tmp_path / "repaired_analysis"
    stale_topic_csv = output_dir / "topic_clustering" / "comment_topics.csv"
    stale_topic_figure = output_dir / "figures" / "core_topic_distribution.png"
    stale_topic_x10_figure = output_dir / "figures" / "core_topic_distribution_x10.png"
    stale_topic_table = output_dir / "tables" / "core_topic_summary.csv"
    stale_topic_table_figure = output_dir / "figures" / "core_topic_summary_table.png"
    for stale_file in (stale_topic_csv, stale_topic_figure, stale_topic_x10_figure, stale_topic_table, stale_topic_table_figure):
        stale_file.parent.mkdir(parents=True, exist_ok=True)
        stale_file.write_text("stale", encoding="utf-8")
    write_comment_csv(cleaned_file, ["AI 工具 提高 效率"])
    write_jsonl(content_file, [content_row(0, [("技术认可", 0.9)])])
    write_jsonl(sentiment_file, [sentiment_row(0, [("乐观", 0.9)], dominant="乐观", polarity="positive")])

    build_repaired_analysis(
        cleaned_file=cleaned_file,
        content_file=content_file,
        sentiment_file=sentiment_file,
        output_dir=output_dir,
        skip_topic_clustering=True,
    )

    assert not stale_topic_csv.exists()
    assert not stale_topic_figure.exists()
    assert not stale_topic_x10_figure.exists()
    assert not stale_topic_table.exists()
    assert not stale_topic_table_figure.exists()


def write_comment_csv(path, contents):
    global LAST_CLEANED_HASHES_BY_INDEX
    LAST_CLEANED_HASHES_BY_INDEX = {}
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["username", "gender", "content", "comment_time", "likes", "ip_location", "signature", "feature"],
        )
        writer.writeheader()
        for index, content in enumerate(contents):
            LAST_CLEANED_HASHES_BY_INDEX[index] = record_dict_hash(record_dict(content, index))
            writer.writerow(record_dict(content, index))


def record_dict(content, index):
    return {
        "username": f"user-{index}",
        "gender": "",
        "content": content,
        "comment_time": "2026-05-10 10:00:00",
        "likes": 1,
        "ip_location": "",
        "signature": "",
        "feature": "aigc",
    }


def content_row(index, labels):
    record = record_dict(["AI 工具 提高 效率", "版权 争议 让 创作者 失望", "普通 无关 评论", "知亿点人工智能适合国内用玫瑰公主浩害羞", "回复 @用户 : 求分享 谢谢 飞吻"][index], index)
    return {
        "record_index": index,
        "record_hash": LAST_CLEANED_HASHES_BY_INDEX[index],
        "record": record,
        "status": "ok",
        "analysis": {
            "summary": "",
            "labels": [{"category": category, "confidence": confidence, "rationale": ""} for category, confidence in labels],
        },
    }


def sentiment_row(index, labels, *, dominant, polarity):
    record = record_dict(["AI 工具 提高 效率", "版权 争议 让 创作者 失望", "普通 无关 评论", "知亿点人工智能适合国内用玫瑰公主浩害羞", "回复 @用户 : 求分享 谢谢 飞吻"][index], index)
    return {
        "record_index": index,
        "record_hash": LAST_CLEANED_HASHES_BY_INDEX[index],
        "record": record,
        "status": "ok",
        "sentiment": {
            "summary": "",
            "dominant_category": dominant,
            "sentiment_polarity": polarity,
            "labels": [{"category": category, "confidence": confidence, "rationale": ""} for category, confidence in labels],
        },
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def count_by_category(rows):
    return {row["category"]: row["count"] for row in rows}


def count_by_field(rows, field):
    return {row[field]: row["count"] for row in rows}


def quality_value(rows, metric):
    row = next(row for row in rows if row["metric"] == metric)
    return row["value"]


def assert_table_headers_are_clean(rows):
    assert rows
    for field in rows[0]:
        assert_clean_text(field)
        assert not field.startswith("raw_")


def assert_clean_text(text):
    normalized = text.lower().replace("_", " ")
    assert "x10" not in normalized
    assert "rawdata" not in normalized.replace(" ", "")
    assert "raw data" not in normalized
