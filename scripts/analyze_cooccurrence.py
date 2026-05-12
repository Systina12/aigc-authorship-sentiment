from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


DEFAULT_INPUT_FILE = Path("data/content_analysis/comment_labels.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/cooccurrence_analysis")

FALLBACK_CATEGORY = "无法归类/无关讨论"
NODE_FILE_NAME = "label_nodes.csv"
EDGE_FILE_NAME = "label_edges.csv"
GRAPH_FILE_NAME = "label_graph.graphml"
SUMMARY_FILE_NAME = "summary.json"


@dataclass(frozen=True)
class CooccurrenceOutputs:
    node_file: Path
    edge_file: Path
    graph_file: Path
    summary_file: Path


@dataclass(frozen=True)
class CooccurrenceReport:
    input_rows: int
    ok_rows: int
    used_rows: int
    skipped_rows: int
    node_count: int
    edge_count: int
    outputs: CooccurrenceOutputs


def analyze_cooccurrence(
    input_file: str | Path = DEFAULT_INPUT_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    limit: int | None = None,
    min_confidence: float = 0.0,
    min_edge_weight: int = 1,
    include_fallback: bool = False,
) -> CooccurrenceReport:
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to 0")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0.0 and 1.0")
    if min_edge_weight < 1:
        raise ValueError("min_edge_weight must be greater than or equal to 1")

    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    outputs = CooccurrenceOutputs(
        node_file=output_path / NODE_FILE_NAME,
        edge_file=output_path / EDGE_FILE_NAME,
        graph_file=output_path / GRAPH_FILE_NAME,
        summary_file=output_path / SUMMARY_FILE_NAME,
    )

    graph = nx.Graph()
    label_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()
    input_rows = 0
    ok_rows = 0
    used_rows = 0

    for row in iter_jsonl(input_path):
        if limit is not None and input_rows >= limit:
            break
        input_rows += 1

        if row.get("status") != "ok":
            continue
        ok_rows += 1

        labels = extract_labels(row, min_confidence=min_confidence, include_fallback=include_fallback)
        if not labels:
            continue

        used_rows += 1
        for label in labels:
            label_counts[label] += 1
        for source, target in itertools.combinations(sorted(labels), 2):
            edge_counts[(source, target)] += 1

    for label, count in label_counts.items():
        graph.add_node(label, count=count)
    for (source, target), weight in edge_counts.items():
        if weight >= min_edge_weight:
            graph.add_edge(source, target, weight=weight)

    annotate_graph(graph)
    write_nodes(graph, outputs.node_file)
    write_edges(graph, outputs.edge_file)
    nx.write_graphml(graph, outputs.graph_file)
    write_summary(
        graph,
        outputs.summary_file,
        input_file=input_path,
        input_rows=input_rows,
        ok_rows=ok_rows,
        used_rows=used_rows,
        min_confidence=min_confidence,
        min_edge_weight=min_edge_weight,
        include_fallback=include_fallback,
        limit=limit,
    )

    return CooccurrenceReport(
        input_rows=input_rows,
        ok_rows=ok_rows,
        used_rows=used_rows,
        skipped_rows=input_rows - used_rows,
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        outputs=outputs,
    )


def iter_jsonl(input_file: Path):
    with input_file.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if not line.strip():
                continue
            yield json.loads(line)


def extract_labels(row: dict[str, Any], *, min_confidence: float, include_fallback: bool) -> list[str]:
    labels = row.get("analysis", {}).get("labels", [])
    if not isinstance(labels, list):
        return []

    selected: dict[str, float] = {}
    for label in labels:
        if not isinstance(label, dict):
            continue

        category = label.get("category", label.get("label"))
        confidence = label.get("confidence", 1.0)
        if not isinstance(category, str):
            continue
        if category == FALLBACK_CATEGORY and not include_fallback:
            continue
        if not isinstance(confidence, int | float) or confidence < min_confidence:
            continue

        selected[category] = max(float(confidence), selected.get(category, 0.0))

    return sorted(selected)


def annotate_graph(graph: nx.Graph) -> None:
    betweenness = nx.betweenness_centrality(graph, weight="weight") if graph.number_of_nodes() else {}
    degree_centrality = nx.degree_centrality(graph) if graph.number_of_nodes() else {}
    communities = detect_communities(graph)

    for node in graph.nodes:
        weighted_degree = sum(data.get("weight", 1) for _, _, data in graph.edges(node, data=True))
        graph.nodes[node]["degree"] = int(graph.degree(node))
        graph.nodes[node]["weighted_degree"] = float(weighted_degree)
        graph.nodes[node]["degree_centrality"] = float(degree_centrality.get(node, 0.0))
        graph.nodes[node]["betweenness_centrality"] = float(betweenness.get(node, 0.0))
        graph.nodes[node]["community"] = int(communities.get(node, -1))


def detect_communities(graph: nx.Graph) -> dict[str, int]:
    if graph.number_of_nodes() < 3 or graph.number_of_edges() < 2:
        return {node: -1 for node in graph.nodes}

    communities = list(greedy_modularity_communities(graph, weight="weight"))
    community_by_node: dict[str, int] = {}
    for community_index, community in enumerate(communities):
        for node in community:
            community_by_node[str(node)] = community_index

    return community_by_node


def write_nodes(graph: nx.Graph, output_file: Path) -> None:
    fieldnames = [
        "label",
        "count",
        "degree",
        "weighted_degree",
        "degree_centrality",
        "betweenness_centrality",
        "community",
    ]
    with output_file.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for node, data in sorted(graph.nodes(data=True), key=lambda item: (-item[1].get("count", 0), item[0])):
            writer.writerow(
                {
                    "label": node,
                    "count": data.get("count", 0),
                    "degree": data.get("degree", 0),
                    "weighted_degree": data.get("weighted_degree", 0.0),
                    "degree_centrality": data.get("degree_centrality", 0.0),
                    "betweenness_centrality": data.get("betweenness_centrality", 0.0),
                    "community": data.get("community", -1),
                }
            )


def write_edges(graph: nx.Graph, output_file: Path) -> None:
    fieldnames = ["source", "target", "weight"]
    with output_file.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for source, target, data in sorted(
            graph.edges(data=True), key=lambda item: (-item[2].get("weight", 0), item[0], item[1])
        ):
            writer.writerow({"source": source, "target": target, "weight": data.get("weight", 1)})


def write_summary(
    graph: nx.Graph,
    output_file: Path,
    *,
    input_file: Path,
    input_rows: int,
    ok_rows: int,
    used_rows: int,
    min_confidence: float,
    min_edge_weight: int,
    include_fallback: bool,
    limit: int | None,
) -> None:
    top_edges = [
        {"source": source, "target": target, "weight": data.get("weight", 1)}
        for source, target, data in sorted(
            graph.edges(data=True), key=lambda item: (-item[2].get("weight", 0), item[0], item[1])
        )[:10]
    ]
    top_nodes = [
        {"label": node, "count": data.get("count", 0), "weighted_degree": data.get("weighted_degree", 0.0)}
        for node, data in sorted(graph.nodes(data=True), key=lambda item: (-item[1].get("count", 0), item[0]))[:10]
    ]
    summary = {
        "input_file": str(input_file),
        "input_rows": input_rows,
        "ok_rows": ok_rows,
        "used_rows": used_rows,
        "skipped_rows": input_rows - used_rows,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "top_nodes": top_nodes,
        "top_edges": top_edges,
        "parameters": {
            "limit": limit,
            "min_confidence": min_confidence,
            "min_edge_weight": min_edge_weight,
            "include_fallback": include_fallback,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a NetworkX co-occurrence graph from content labels.")
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT_FILE),
        help="Content-analysis JSONL file. Defaults to data/content_analysis/comment_labels.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder for co-occurrence outputs. Defaults to data/cooccurrence_analysis.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only consider the first N input rows.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only include labels with at least this confidence.",
    )
    parser.add_argument(
        "--min-edge-weight",
        type=int,
        default=1,
        help="Only keep edges with at least this co-occurrence count.",
    )
    parser.add_argument(
        "--include-fallback",
        action="store_true",
        help="Include fallback labels such as 无法归类/无关讨论.",
    )
    args = parser.parse_args()

    report = analyze_cooccurrence(
        input_file=args.input_file,
        output_dir=args.output_dir,
        limit=args.limit,
        min_confidence=args.min_confidence,
        min_edge_weight=args.min_edge_weight,
        include_fallback=args.include_fallback,
    )
    print(f"Input rows: {report.input_rows}")
    print(f"OK rows: {report.ok_rows}")
    print(f"Used rows: {report.used_rows}")
    print(f"Skipped rows: {report.skipped_rows}")
    print(f"Nodes: {report.node_count}")
    print(f"Edges: {report.edge_count}")
    print(f"Node file: {report.outputs.node_file}")
    print(f"Edge file: {report.outputs.edge_file}")
    print(f"Graph file: {report.outputs.graph_file}")
    print(f"Summary file: {report.outputs.summary_file}")


if __name__ == "__main__":
    main()
