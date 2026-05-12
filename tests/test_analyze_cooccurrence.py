import csv
import json

import networkx as nx
import pytest

from scripts.analyze_cooccurrence import analyze_cooccurrence


def test_analyze_cooccurrence_writes_nodes_edges_graph_and_summary(tmp_path):
    input_file = tmp_path / "comment_labels.jsonl"
    output_dir = tmp_path / "cooccurrence"
    write_jsonl(
        input_file,
        [
            ok_row(["技术认可", "工具化认知"]),
            ok_row(["技术认可", "工具化认知", "职业焦虑"]),
            ok_row(["技术认可"]),
            ok_row(["无法归类/无关讨论"]),
            {"status": "error", "analysis": None},
        ],
    )

    report = analyze_cooccurrence(input_file=input_file, output_dir=output_dir)

    nodes = read_csv(output_dir / "label_nodes.csv")
    edges = read_csv(output_dir / "label_edges.csv")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    graph = nx.read_graphml(output_dir / "label_graph.graphml")

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
    assert summary["node_count"] == 3
    assert summary["edge_count"] == 3
    assert summary["top_edges"][0] == {"source": "工具化认知", "target": "技术认可", "weight": 2}
    assert summary["parameters"]["include_fallback"] is False


def test_analyze_cooccurrence_filters_by_confidence_edge_weight_and_limit(tmp_path):
    input_file = tmp_path / "comment_labels.jsonl"
    output_dir = tmp_path / "cooccurrence"
    write_jsonl(
        input_file,
        [
            ok_row(
                [
                    ("技术认可", 0.95),
                    ("工具化认知", 0.85),
                    ("职业焦虑", 0.4),
                ]
            ),
            ok_row([("技术认可", 0.9), ("工具化认知", 0.9)]),
            ok_row([("版权争议", 1.0), ("职业焦虑", 1.0)]),
        ],
    )

    report = analyze_cooccurrence(
        input_file=input_file,
        output_dir=output_dir,
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


def test_analyze_cooccurrence_can_include_fallback(tmp_path):
    input_file = tmp_path / "comment_labels.jsonl"
    output_dir = tmp_path / "cooccurrence"
    write_jsonl(input_file, [ok_row(["无法归类/无关讨论"])])

    report = analyze_cooccurrence(input_file=input_file, output_dir=output_dir, include_fallback=True)

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

    with pytest.raises(ValueError, match="min_edge_weight"):
        analyze_cooccurrence(input_file=input_file, output_dir=tmp_path / "out", min_edge_weight=0)


def ok_row(labels):
    normalized_labels = []
    for label in labels:
        if isinstance(label, tuple):
            category, confidence = label
        else:
            category, confidence = label, 1.0
        normalized_labels.append({"category": category, "confidence": confidence, "rationale": "test"})

    return {
        "status": "ok",
        "analysis": {
            "summary": "test",
            "labels": normalized_labels,
        },
    }


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def node_counts(rows):
    return {row["label"]: row["count"] for row in rows}


def edge_weights(rows):
    return {frozenset((row["source"], row["target"])): row["weight"] for row in rows}
