import csv
import json

import networkx as nx
import pytest

from scripts.analyze_cooccurrence import analyze_cooccurrence


def test_analyze_cooccurrence_writes_content_nodes_edges_graph_and_summary(tmp_path):
    input_file = tmp_path / "comment_labels.jsonl"
    output_dir = tmp_path / "cooccurrence"
    write_jsonl(
        input_file,
        [
            content_row(0, "hash-0", ["技术认可", "工具化认知"]),
            content_row(1, "hash-1", ["技术认可", "工具化认知", "职业焦虑"]),
            content_row(2, "hash-2", ["技术认可"]),
            content_row(3, "hash-3", ["无法归类/无关讨论"]),
            {"status": "error", "analysis": None},
        ],
    )

    report = analyze_cooccurrence(input_file=input_file, output_dir=output_dir, sentiment_file=None, topic_file=None)

    nodes = read_csv(output_dir / "label_nodes.csv")
    edges = read_csv(output_dir / "label_edges.csv")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    graph = nx.read_graphml(output_dir / "label_graph.graphml")
    content_graph = nx.read_graphml(output_dir / "content_label_graph.graphml")

    assert report.input_rows == 5
    assert report.ok_rows == 4
    assert report.used_rows == 3
    assert report.node_count == 3
    assert report.edge_count == 3
    assert node_counts(nodes) == {"技术认可": "3", "工具化认知": "2", "职业焦虑": "1"}
    assert edge_weights(edges) == {
        frozenset(("工具化认知", "技术认可")): "2",
        frozenset(("职业焦虑", "工具化认知")): "1",
        frozenset(("职业焦虑", "技术认可")): "1",
    }
    assert "无法归类/无关讨论" not in graph.nodes
    assert graph["工具化认知"]["技术认可"]["weight"] == 2
    assert content_graph["工具化认知"]["技术认可"]["weight"] == 2
    assert summary["node_count"] == 3
    assert summary["edge_count"] == 3
    assert summary["top_edges"][0] == {"source": "工具化认知", "target": "技术认可", "weight": 2}
    assert summary["parameters"]["include_fallback"] is False
    assert summary["networks"]["content_labels"]["edge_count"] == 3


def test_analyze_cooccurrence_builds_sentiment_and_cross_networks(tmp_path):
    content_file = tmp_path / "comment_labels.jsonl"
    sentiment_file = tmp_path / "comment_sentiment.jsonl"
    output_dir = tmp_path / "cooccurrence"
    write_jsonl(
        content_file,
        [
            content_row(0, "hash-0", ["职业焦虑", "版权争议"]),
            content_row(1, "hash-1", ["职业焦虑"]),
            content_row(2, "hash-2", ["技术认可"]),
        ],
    )
    write_jsonl(
        sentiment_file,
        [
            sentiment_row(0, "hash-0", ["焦虑", "愤怒"], dominant="焦虑", polarity="negative"),
            sentiment_row(1, "hash-1", ["焦虑"], dominant="焦虑", polarity="negative"),
            sentiment_row(2, "hash-2", ["乐观"], dominant="乐观", polarity="positive"),
            sentiment_row(3, "hash-3", ["中性/无法判断"], dominant="中性/无法判断", polarity="neutral"),
            {"status": "error", "sentiment": None},
        ],
    )

    report = analyze_cooccurrence(
        input_file=content_file,
        output_dir=output_dir,
        sentiment_file=sentiment_file,
        topic_file=None,
    )

    sentiment_edges = read_csv(output_dir / "sentiment_label_edges.csv")
    content_sentiment_edges = read_csv(output_dir / "content_sentiment_edges.csv")
    content_polarity_edges = read_csv(output_dir / "content_polarity_edges.csv")
    sentiment_polarity_edges = read_csv(output_dir / "sentiment_polarity_edges.csv")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    cross_graph = nx.read_graphml(output_dir / "content_sentiment_graph.graphml")

    assert report.network_reports["sentiment_labels"].used_rows == 3
    assert edge_weights(sentiment_edges) == {frozenset(("sentiment:愤怒", "sentiment:焦虑")): "1"}
    assert edge_weights(content_sentiment_edges) == {
        frozenset(("职业焦虑", "sentiment:焦虑")): "2",
        frozenset(("版权争议", "sentiment:焦虑")): "1",
        frozenset(("版权争议", "sentiment:愤怒")): "1",
        frozenset(("职业焦虑", "sentiment:愤怒")): "1",
        frozenset(("技术认可", "sentiment:乐观")): "1",
    }
    assert edge_weights(content_polarity_edges) == {
        frozenset(("职业焦虑", "polarity:negative")): "2",
        frozenset(("版权争议", "polarity:negative")): "1",
        frozenset(("技术认可", "polarity:positive")): "1",
    }
    assert edge_weights(sentiment_polarity_edges) == {
        frozenset(("sentiment:焦虑", "polarity:negative")): "2",
        frozenset(("sentiment:愤怒", "polarity:negative")): "1",
        frozenset(("sentiment:乐观", "polarity:positive")): "1",
    }
    assert cross_graph["职业焦虑"]["sentiment:焦虑"]["weight"] == 2
    assert "sentiment:中性/无法判断" not in cross_graph.nodes
    assert summary["optional_inputs"]["sentiment_status"] == "loaded"
    assert summary["networks"]["content_sentiment"]["edge_count"] == 5


def test_analyze_cooccurrence_filters_sentiment_confidence_and_neutral_polarity(tmp_path):
    content_file = tmp_path / "comment_labels.jsonl"
    sentiment_file = tmp_path / "comment_sentiment.jsonl"
    output_dir = tmp_path / "cooccurrence"
    write_jsonl(
        content_file,
        [
            content_row(0, "hash-0", ["技术认可"]),
            content_row(1, "hash-1", ["工具化认知"]),
        ],
    )
    write_jsonl(
        sentiment_file,
        [
            sentiment_row(0, "hash-0", [("乐观", 0.9), ("质疑", 0.3)], dominant="乐观", polarity="positive"),
            sentiment_row(1, "hash-1", [("中性/无法判断", 1.0)], dominant="中性/无法判断", polarity="neutral"),
        ],
    )

    analyze_cooccurrence(
        input_file=content_file,
        output_dir=output_dir,
        sentiment_file=sentiment_file,
        topic_file=None,
        min_sentiment_confidence=0.8,
        include_neutral_polarity=False,
    )

    content_sentiment_edges = read_csv(output_dir / "content_sentiment_edges.csv")
    content_polarity_edges = read_csv(output_dir / "content_polarity_edges.csv")

    assert edge_weights(content_sentiment_edges) == {frozenset(("技术认可", "sentiment:乐观")): "1"}
    assert edge_weights(content_polarity_edges) == {frozenset(("技术认可", "polarity:positive")): "1"}


def test_analyze_cooccurrence_builds_topic_cross_networks_when_topic_file_exists(tmp_path):
    content_file = tmp_path / "comment_labels.jsonl"
    sentiment_file = tmp_path / "comment_sentiment.jsonl"
    topic_file = tmp_path / "comment_topics.csv"
    output_dir = tmp_path / "cooccurrence"
    write_jsonl(
        content_file,
        [
            content_row(0, "hash-0", ["技术认可"]),
            content_row(1, "hash-1", ["版权争议"]),
            content_row(2, "hash-2", ["职业焦虑"]),
        ],
    )
    write_jsonl(
        sentiment_file,
        [
            sentiment_row(0, "hash-0", ["乐观"], dominant="乐观", polarity="positive"),
            sentiment_row(1, "hash-1", ["愤怒"], dominant="愤怒", polarity="negative"),
            sentiment_row(2, "hash-2", ["焦虑"], dominant="焦虑", polarity="negative"),
        ],
    )
    write_topic_csv(
        topic_file,
        [
            {"record_index": "0", "topic": "0", "topic_probability": "0.91"},
            {"record_index": "1", "topic": "1", "topic_probability": "0.82"},
            {"record_index": "2", "topic": "-1", "topic_probability": "0.2"},
        ],
    )
    write_topic_info_csv(topic_file.with_name("topic_info.csv"))

    analyze_cooccurrence(
        input_file=content_file,
        output_dir=output_dir,
        sentiment_file=sentiment_file,
        topic_file=topic_file,
        min_topic_probability=0.5,
    )

    topic_content_edges = read_csv(output_dir / "topic_content_edges.csv")
    topic_sentiment_edges = read_csv(output_dir / "topic_sentiment_edges.csv")
    topic_polarity_edges = read_csv(output_dir / "topic_polarity_edges.csv")
    topic_graph = nx.read_graphml(output_dir / "topic_content_graph.graphml")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert edge_weights(topic_content_edges) == {
        frozenset(("topic:0", "技术认可")): "1",
        frozenset(("topic:1", "版权争议")): "1",
    }
    assert edge_weights(topic_sentiment_edges) == {
        frozenset(("topic:0", "sentiment:乐观")): "1",
        frozenset(("topic:1", "sentiment:愤怒")): "1",
    }
    assert edge_weights(topic_polarity_edges) == {
        frozenset(("topic:0", "polarity:positive")): "1",
        frozenset(("topic:1", "polarity:negative")): "1",
    }
    assert topic_graph.nodes["topic:0"]["label"].startswith("Topic 0: ai_tools")
    assert "topic:-1" not in topic_graph.nodes
    assert summary["optional_inputs"]["topic_status"] == "loaded"
    assert summary["networks"]["topic_content"]["used_rows"] == 2


def test_analyze_cooccurrence_filters_by_confidence_edge_weight_and_limit(tmp_path):
    input_file = tmp_path / "comment_labels.jsonl"
    output_dir = tmp_path / "cooccurrence"
    write_jsonl(
        input_file,
        [
            content_row(
                0,
                "hash-0",
                [
                    ("技术认可", 0.95),
                    ("工具化认知", 0.85),
                    ("职业焦虑", 0.4),
                ],
            ),
            content_row(1, "hash-1", [("技术认可", 0.9), ("工具化认知", 0.9)]),
            content_row(2, "hash-2", [("版权争议", 1.0), ("职业焦虑", 1.0)]),
        ],
    )

    report = analyze_cooccurrence(
        input_file=input_file,
        output_dir=output_dir,
        sentiment_file=None,
        topic_file=None,
        limit=2,
        min_confidence=0.8,
        min_edge_weight=2,
    )

    nodes = read_csv(output_dir / "label_nodes.csv")
    edges = read_csv(output_dir / "label_edges.csv")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert report.input_rows == 2
    assert report.used_rows == 2
    assert report.node_count == 2
    assert report.edge_count == 1
    assert node_counts(nodes) == {"技术认可": "2", "工具化认知": "2"}
    assert edge_weights(edges) == {frozenset(("工具化认知", "技术认可")): "2"}
    assert summary["parameters"]["limit"] == 2
    assert summary["parameters"]["min_confidence"] == 0.8
    assert summary["parameters"]["min_edge_weight"] == 2


def test_analyze_cooccurrence_can_include_content_fallback(tmp_path):
    input_file = tmp_path / "comment_labels.jsonl"
    output_dir = tmp_path / "cooccurrence"
    write_jsonl(input_file, [content_row(0, "hash-0", ["无法归类/无关讨论"])])

    report = analyze_cooccurrence(
        input_file=input_file,
        output_dir=output_dir,
        sentiment_file=None,
        topic_file=None,
        include_fallback=True,
    )

    nodes = read_csv(output_dir / "label_nodes.csv")
    edges = read_csv(output_dir / "label_edges.csv")

    assert report.node_count == 1
    assert report.edge_count == 0
    assert node_counts(nodes) == {"无法归类/无关讨论": "1"}
    assert edges == []


def test_analyze_cooccurrence_validates_parameters(tmp_path):
    input_file = tmp_path / "comment_labels.jsonl"
    write_jsonl(input_file, [])

    with pytest.raises(ValueError, match="limit"):
        analyze_cooccurrence(input_file=input_file, output_dir=tmp_path / "out", limit=-1)

    with pytest.raises(ValueError, match="min_confidence"):
        analyze_cooccurrence(input_file=input_file, output_dir=tmp_path / "out", min_confidence=1.5)

    with pytest.raises(ValueError, match="min_sentiment_confidence"):
        analyze_cooccurrence(input_file=input_file, output_dir=tmp_path / "out", min_sentiment_confidence=-0.1)

    with pytest.raises(ValueError, match="min_topic_probability"):
        analyze_cooccurrence(input_file=input_file, output_dir=tmp_path / "out", min_topic_probability=2)

    with pytest.raises(ValueError, match="min_edge_weight"):
        analyze_cooccurrence(input_file=input_file, output_dir=tmp_path / "out", min_edge_weight=0)


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


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


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


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def node_counts(rows):
    return {row["label"]: row["count"] for row in rows}


def edge_weights(rows):
    return {frozenset((row["source"], row["target"])): row["weight"] for row in rows}
