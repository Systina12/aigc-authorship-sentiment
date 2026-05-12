from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


DEFAULT_INPUT_FILE = Path("data/content_analysis/comment_labels.jsonl")
DEFAULT_SENTIMENT_FILE = Path("data/sentiment_analysis/comment_sentiment.jsonl")
DEFAULT_TOPIC_FILE = Path("data/topic_clustering/comment_topics.csv")
DEFAULT_OUTPUT_DIR = Path("data/cooccurrence_analysis")

CONTENT_FALLBACK_CATEGORY = "无法归类/无关讨论"
SENTIMENT_FALLBACK_CATEGORY = "中性/无法判断"
NEUTRAL_POLARITY = "neutral"
NOISE_TOPIC = "-1"

NODE_FILE_NAME = "label_nodes.csv"
EDGE_FILE_NAME = "label_edges.csv"
GRAPH_FILE_NAME = "label_graph.graphml"
SUMMARY_FILE_NAME = "summary.json"

NETWORK_FILE_PREFIXES = {
    "content_labels": "content_label",
    "sentiment_labels": "sentiment_label",
    "content_sentiment": "content_sentiment",
    "content_dominant_sentiment": "content_dominant_sentiment",
    "content_polarity": "content_polarity",
    "sentiment_polarity": "sentiment_polarity",
    "topic_content": "topic_content",
    "topic_sentiment": "topic_sentiment",
    "topic_polarity": "topic_polarity",
}


@dataclass(frozen=True)
class CooccurrenceOutputs:
    node_file: Path
    edge_file: Path
    graph_file: Path
    summary_file: Path
    network_files: dict[str, "NetworkOutputs"] = field(default_factory=dict)


@dataclass(frozen=True)
class NetworkOutputs:
    node_file: Path
    edge_file: Path
    graph_file: Path


@dataclass(frozen=True)
class NetworkReport:
    used_rows: int
    node_count: int
    edge_count: int
    outputs: NetworkOutputs


@dataclass(frozen=True)
class CooccurrenceReport:
    input_rows: int
    ok_rows: int
    used_rows: int
    skipped_rows: int
    node_count: int
    edge_count: int
    outputs: CooccurrenceOutputs
    network_reports: dict[str, NetworkReport] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeOccurrence:
    node_id: str
    label: str
    node_type: str


@dataclass(frozen=True)
class LabelRecord:
    record_index: int | None
    record_hash: str | None
    labels: tuple[str, ...] = ()
    dominant_category: str | None = None
    sentiment_polarity: str | None = None


@dataclass(frozen=True)
class TopicRecord:
    record_index: int
    topic: str
    probability: float | None
    label: str


@dataclass(frozen=True)
class JsonlLoadResult:
    input_rows: int
    ok_rows: int
    records: list[LabelRecord]


def analyze_cooccurrence(
    input_file: str | Path = DEFAULT_INPUT_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    sentiment_file: str | Path | None = DEFAULT_SENTIMENT_FILE,
    topic_file: str | Path | None = DEFAULT_TOPIC_FILE,
    limit: int | None = None,
    min_confidence: float = 0.0,
    min_sentiment_confidence: float | None = None,
    min_topic_probability: float = 0.0,
    min_edge_weight: int = 1,
    include_fallback: bool = False,
    include_sentiment_fallback: bool = False,
    include_neutral_polarity: bool = True,
    include_topic_noise: bool = False,
) -> CooccurrenceReport:
    validate_parameters(
        limit=limit,
        min_confidence=min_confidence,
        min_sentiment_confidence=min_sentiment_confidence,
        min_topic_probability=min_topic_probability,
        min_edge_weight=min_edge_weight,
    )

    content_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    network_outputs = {name: network_output_paths(output_path, prefix) for name, prefix in NETWORK_FILE_PREFIXES.items()}
    outputs = CooccurrenceOutputs(
        node_file=output_path / NODE_FILE_NAME,
        edge_file=output_path / EDGE_FILE_NAME,
        graph_file=output_path / GRAPH_FILE_NAME,
        summary_file=output_path / SUMMARY_FILE_NAME,
        network_files=network_outputs,
    )

    sentiment_threshold = min_confidence if min_sentiment_confidence is None else min_sentiment_confidence
    content_data = load_content_records(
        content_path,
        limit=limit,
        min_confidence=min_confidence,
        include_fallback=include_fallback,
    )
    content_records = content_data.records
    content_graph, content_used_rows = build_within_network(
        (content_nodes(record) for record in content_records),
        min_edge_weight=min_edge_weight,
    )
    write_network(content_graph, outputs.node_file, outputs.edge_file, outputs.graph_file)
    write_network(
        content_graph,
        network_outputs["content_labels"].node_file,
        network_outputs["content_labels"].edge_file,
        network_outputs["content_labels"].graph_file,
    )

    graphs: dict[str, tuple[nx.Graph, int]] = {"content_labels": (content_graph, content_used_rows)}
    optional_inputs: dict[str, Any] = {
        "sentiment_file": None,
        "sentiment_status": "disabled",
        "topic_file": None,
        "topic_status": "disabled",
    }

    sentiment_records: list[LabelRecord] = []
    if sentiment_file is not None:
        sentiment_path = Path(sentiment_file)
        optional_inputs["sentiment_file"] = str(sentiment_path)
        if sentiment_path.exists():
            sentiment_data = load_sentiment_records(
                sentiment_path,
                limit=limit,
                min_confidence=sentiment_threshold,
                include_fallback=include_sentiment_fallback,
            )
            sentiment_records = sentiment_data.records
            optional_inputs["sentiment_status"] = "loaded"
            optional_inputs["sentiment_input_rows"] = sentiment_data.input_rows
            optional_inputs["sentiment_ok_rows"] = sentiment_data.ok_rows

            sentiment_graph, sentiment_used_rows = build_within_network(
                (sentiment_nodes(record, include_fallback=include_sentiment_fallback) for record in sentiment_records),
                min_edge_weight=min_edge_weight,
            )
            graphs["sentiment_labels"] = (sentiment_graph, sentiment_used_rows)

            matched_records = match_label_records(content_records, sentiment_records)
            add_cross_network(
                graphs,
                "content_sentiment",
                (
                    (content_nodes(content), sentiment_nodes(sentiment, include_fallback=include_sentiment_fallback))
                    for content, sentiment in matched_records
                ),
                min_edge_weight=min_edge_weight,
            )
            add_cross_network(
                graphs,
                "content_dominant_sentiment",
                (
                    (content_nodes(content), dominant_sentiment_nodes(sentiment))
                    for content, sentiment in matched_records
                ),
                min_edge_weight=min_edge_weight,
            )
            add_cross_network(
                graphs,
                "content_polarity",
                (
                    (content_nodes(content), polarity_nodes(sentiment, include_neutral=include_neutral_polarity))
                    for content, sentiment in matched_records
                ),
                min_edge_weight=min_edge_weight,
            )
            add_cross_network(
                graphs,
                "sentiment_polarity",
                (
                    (
                        sentiment_nodes(sentiment, include_fallback=include_sentiment_fallback),
                        polarity_nodes(sentiment, include_neutral=include_neutral_polarity),
                    )
                    for sentiment in sentiment_records
                ),
                min_edge_weight=min_edge_weight,
            )
        else:
            optional_inputs["sentiment_status"] = "missing"

    if topic_file is not None:
        topic_path = Path(topic_file)
        optional_inputs["topic_file"] = str(topic_path)
        if topic_path.exists():
            topic_records = load_topic_records(
                topic_path,
                limit=limit,
                min_probability=min_topic_probability,
                include_noise=include_topic_noise,
            )
            optional_inputs["topic_status"] = "loaded"
            optional_inputs["topic_rows"] = len(topic_records)
            topics_by_index = {record.record_index: record for record in topic_records}
            content_by_index = records_by_index(content_records)
            matched_topic_content = [
                (topics_by_index[index], content_by_index[index])
                for index in sorted(topics_by_index.keys() & content_by_index.keys())
            ]
            add_cross_network(
                graphs,
                "topic_content",
                ((topic_nodes(topic), content_nodes(content)) for topic, content in matched_topic_content),
                min_edge_weight=min_edge_weight,
            )

            if sentiment_records:
                sentiment_by_index = records_by_index(sentiment_records)
                matched_topic_sentiment = [
                    (topics_by_index[index], sentiment_by_index[index])
                    for index in sorted(topics_by_index.keys() & sentiment_by_index.keys())
                ]
                add_cross_network(
                    graphs,
                    "topic_sentiment",
                    (
                        (topic_nodes(topic), sentiment_nodes(sentiment, include_fallback=include_sentiment_fallback))
                        for topic, sentiment in matched_topic_sentiment
                    ),
                    min_edge_weight=min_edge_weight,
                )
                add_cross_network(
                    graphs,
                    "topic_polarity",
                    (
                        (topic_nodes(topic), polarity_nodes(sentiment, include_neutral=include_neutral_polarity))
                        for topic, sentiment in matched_topic_sentiment
                    ),
                    min_edge_weight=min_edge_weight,
                )
        else:
            optional_inputs["topic_status"] = "missing"

    network_reports: dict[str, NetworkReport] = {}
    for name, (graph, used_rows) in graphs.items():
        network_file = network_outputs[name]
        if name != "content_labels":
            write_network(graph, network_file.node_file, network_file.edge_file, network_file.graph_file)
        network_reports[name] = NetworkReport(
            used_rows=used_rows,
            node_count=graph.number_of_nodes(),
            edge_count=graph.number_of_edges(),
            outputs=network_file,
        )

    write_summary(
        content_graph,
        outputs.summary_file,
        input_file=content_path,
        input_rows=content_data.input_rows,
        ok_rows=content_data.ok_rows,
        used_rows=content_used_rows,
        min_confidence=min_confidence,
        min_sentiment_confidence=sentiment_threshold,
        min_topic_probability=min_topic_probability,
        min_edge_weight=min_edge_weight,
        include_fallback=include_fallback,
        include_sentiment_fallback=include_sentiment_fallback,
        include_neutral_polarity=include_neutral_polarity,
        include_topic_noise=include_topic_noise,
        limit=limit,
        graphs=graphs,
        optional_inputs=optional_inputs,
    )

    return CooccurrenceReport(
        input_rows=content_data.input_rows,
        ok_rows=content_data.ok_rows,
        used_rows=content_used_rows,
        skipped_rows=content_data.input_rows - content_used_rows,
        node_count=content_graph.number_of_nodes(),
        edge_count=content_graph.number_of_edges(),
        outputs=outputs,
        network_reports=network_reports,
    )


def validate_parameters(
    *,
    limit: int | None,
    min_confidence: float,
    min_sentiment_confidence: float | None,
    min_topic_probability: float,
    min_edge_weight: int,
) -> None:
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to 0")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0.0 and 1.0")
    if min_sentiment_confidence is not None and not 0 <= min_sentiment_confidence <= 1:
        raise ValueError("min_sentiment_confidence must be between 0.0 and 1.0")
    if not 0 <= min_topic_probability <= 1:
        raise ValueError("min_topic_probability must be between 0.0 and 1.0")
    if min_edge_weight < 1:
        raise ValueError("min_edge_weight must be greater than or equal to 1")


def network_output_paths(output_dir: Path, prefix: str) -> NetworkOutputs:
    return NetworkOutputs(
        node_file=output_dir / f"{prefix}_nodes.csv",
        edge_file=output_dir / f"{prefix}_edges.csv",
        graph_file=output_dir / f"{prefix}_graph.graphml",
    )


def load_content_records(
    input_file: Path,
    *,
    limit: int | None,
    min_confidence: float,
    include_fallback: bool,
) -> JsonlLoadResult:
    return load_label_records(
        input_file,
        limit=limit,
        section_name="analysis",
        min_confidence=min_confidence,
        fallback_category=CONTENT_FALLBACK_CATEGORY,
        include_fallback=include_fallback,
    )


def load_sentiment_records(
    input_file: Path,
    *,
    limit: int | None,
    min_confidence: float,
    include_fallback: bool,
) -> JsonlLoadResult:
    return load_label_records(
        input_file,
        limit=limit,
        section_name="sentiment",
        min_confidence=min_confidence,
        fallback_category=SENTIMENT_FALLBACK_CATEGORY,
        include_fallback=include_fallback,
    )


def load_label_records(
    input_file: Path,
    *,
    limit: int | None,
    section_name: str,
    min_confidence: float,
    fallback_category: str,
    include_fallback: bool,
) -> JsonlLoadResult:
    input_rows = 0
    ok_rows = 0
    records_by_key: dict[str, LabelRecord] = {}

    for row in iter_jsonl(input_file):
        if limit is not None and input_rows >= limit:
            break
        input_rows += 1

        if row.get("status") != "ok":
            continue
        ok_rows += 1

        labels = extract_labels(
            row.get(section_name, {}),
            min_confidence=min_confidence,
            fallback_category=fallback_category,
            include_fallback=include_fallback,
        )
        record = LabelRecord(
            record_index=parse_optional_int(row.get("record_index")),
            record_hash=parse_optional_str(row.get("record_hash")),
            labels=tuple(labels),
            dominant_category=parse_optional_str(row.get(section_name, {}).get("dominant_category"))
            if isinstance(row.get(section_name), dict)
            else None,
            sentiment_polarity=parse_optional_str(row.get(section_name, {}).get("sentiment_polarity"))
            if isinstance(row.get(section_name), dict)
            else None,
        )
        key = record_key(record)
        if key is None:
            continue
        records_by_key[key] = record

    return JsonlLoadResult(input_rows=input_rows, ok_rows=ok_rows, records=list(records_by_key.values()))


def iter_jsonl(input_file: Path) -> Iterable[dict[str, Any]]:
    with input_file.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if not line.strip():
                continue
            yield json.loads(line)


def extract_labels(
    analysis: object,
    *,
    min_confidence: float,
    fallback_category: str,
    include_fallback: bool,
) -> list[str]:
    if not isinstance(analysis, dict):
        return []
    labels = analysis.get("labels", [])
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
        if category == fallback_category and not include_fallback:
            continue
        if not isinstance(confidence, int | float) or confidence < min_confidence:
            continue

        selected[category] = max(float(confidence), selected.get(category, 0.0))

    return sorted(selected)


def load_topic_records(
    topic_file: Path,
    *,
    limit: int | None,
    min_probability: float,
    include_noise: bool,
) -> list[TopicRecord]:
    topic_names = load_topic_names(topic_file.with_name("topic_info.csv"))
    records: list[TopicRecord] = []
    with topic_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            record_index = parse_optional_int(row.get("record_index"))
            if record_index is None:
                continue
            if limit is not None and record_index >= limit:
                continue

            topic = parse_optional_str(row.get("topic"))
            if topic is None:
                continue
            if topic == NOISE_TOPIC and not include_noise:
                continue

            probability = parse_float(row.get("topic_probability"))
            if probability is not None and probability < min_probability:
                continue
            if probability is None and min_probability > 0:
                continue

            topic_label = topic_names.get(topic, f"Topic {topic}")
            records.append(
                TopicRecord(
                    record_index=record_index,
                    topic=topic,
                    probability=probability,
                    label=topic_label,
                )
            )

    return records


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


def build_within_network(
    rows: Iterable[list[NodeOccurrence]],
    *,
    min_edge_weight: int,
) -> tuple[nx.Graph, int]:
    graph = nx.Graph()
    node_counts: Counter[NodeOccurrence] = Counter()
    edge_counts: Counter[tuple[NodeOccurrence, NodeOccurrence]] = Counter()
    used_rows = 0

    for row_nodes in rows:
        unique_nodes = dedupe_nodes(row_nodes)
        if not unique_nodes:
            continue

        used_rows += 1
        for node in unique_nodes:
            node_counts[node] += 1
        for source, target in itertools.combinations(unique_nodes, 2):
            edge_counts[ordered_edge(source, target)] += 1

    add_nodes_and_edges(graph, node_counts, edge_counts, min_edge_weight=min_edge_weight)
    return graph, used_rows


def build_cross_network(
    rows: Iterable[tuple[list[NodeOccurrence], list[NodeOccurrence]]],
    *,
    min_edge_weight: int,
) -> tuple[nx.Graph, int]:
    graph = nx.Graph()
    node_counts: Counter[NodeOccurrence] = Counter()
    edge_counts: Counter[tuple[NodeOccurrence, NodeOccurrence]] = Counter()
    used_rows = 0

    for left_nodes, right_nodes in rows:
        left_unique = dedupe_nodes(left_nodes)
        right_unique = dedupe_nodes(right_nodes)
        if not left_unique or not right_unique:
            continue

        used_rows += 1
        for node in left_unique + right_unique:
            node_counts[node] += 1
        for source in left_unique:
            for target in right_unique:
                if source.node_id == target.node_id:
                    continue
                edge_counts[ordered_edge(source, target)] += 1

    add_nodes_and_edges(graph, node_counts, edge_counts, min_edge_weight=min_edge_weight)
    return graph, used_rows


def add_cross_network(
    graphs: dict[str, tuple[nx.Graph, int]],
    name: str,
    rows: Iterable[tuple[list[NodeOccurrence], list[NodeOccurrence]]],
    *,
    min_edge_weight: int,
) -> None:
    graphs[name] = build_cross_network(rows, min_edge_weight=min_edge_weight)


def add_nodes_and_edges(
    graph: nx.Graph,
    node_counts: Counter[NodeOccurrence],
    edge_counts: Counter[tuple[NodeOccurrence, NodeOccurrence]],
    *,
    min_edge_weight: int,
) -> None:
    for node, count in node_counts.items():
        graph.add_node(node.node_id, label=node.label, node_type=node.node_type, count=count)
    for (source, target), weight in edge_counts.items():
        if weight >= min_edge_weight:
            graph.add_edge(source.node_id, target.node_id, weight=weight)

    annotate_graph(graph)


def dedupe_nodes(nodes: Iterable[NodeOccurrence]) -> list[NodeOccurrence]:
    return sorted({node.node_id: node for node in nodes}.values(), key=lambda node: node.node_id)


def ordered_edge(source: NodeOccurrence, target: NodeOccurrence) -> tuple[NodeOccurrence, NodeOccurrence]:
    return (source, target) if source.node_id <= target.node_id else (target, source)


def content_nodes(record: LabelRecord) -> list[NodeOccurrence]:
    return [NodeOccurrence(label, label, "content_label") for label in record.labels]


def sentiment_nodes(record: LabelRecord, *, include_fallback: bool = False) -> list[NodeOccurrence]:
    return [
        NodeOccurrence(f"sentiment:{label}", label, "sentiment_label")
        for label in record.labels
        if include_fallback or label != SENTIMENT_FALLBACK_CATEGORY
    ]


def dominant_sentiment_nodes(record: LabelRecord) -> list[NodeOccurrence]:
    if not record.dominant_category or record.dominant_category == SENTIMENT_FALLBACK_CATEGORY:
        return []
    return [NodeOccurrence(f"dominant_sentiment:{record.dominant_category}", record.dominant_category, "dominant_sentiment")]


def polarity_nodes(record: LabelRecord, *, include_neutral: bool) -> list[NodeOccurrence]:
    polarity = record.sentiment_polarity
    if not polarity or (polarity == NEUTRAL_POLARITY and not include_neutral):
        return []
    return [NodeOccurrence(f"polarity:{polarity}", polarity, "sentiment_polarity")]


def topic_nodes(record: TopicRecord) -> list[NodeOccurrence]:
    return [NodeOccurrence(f"topic:{record.topic}", record.label, "topic")]


def match_label_records(left_records: list[LabelRecord], right_records: list[LabelRecord]) -> list[tuple[LabelRecord, LabelRecord]]:
    right_by_hash = {record.record_hash: record for record in right_records if record.record_hash}
    right_by_index = records_by_index(right_records)
    matched: list[tuple[LabelRecord, LabelRecord]] = []
    seen_right_keys: set[str] = set()

    for left in left_records:
        right: LabelRecord | None = None
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


def records_by_index(records: Iterable[LabelRecord]) -> dict[int, LabelRecord]:
    return {record.record_index: record for record in records if record.record_index is not None}


def record_key(record: LabelRecord) -> str | None:
    if record.record_hash:
        return f"hash:{record.record_hash}"
    if record.record_index is not None:
        return f"index:{record.record_index}"
    return None


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
        return {str(node): -1 for node in graph.nodes}

    communities = list(greedy_modularity_communities(graph, weight="weight"))
    community_by_node: dict[str, int] = {}
    for community_index, community in enumerate(communities):
        for node in community:
            community_by_node[str(node)] = community_index

    return community_by_node


def write_network(graph: nx.Graph, node_file: Path, edge_file: Path, graph_file: Path) -> None:
    write_nodes(graph, node_file)
    write_edges(graph, edge_file)
    nx.write_graphml(graph, graph_file)


def write_nodes(graph: nx.Graph, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "node_id",
        "node_type",
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
        for node, data in sorted(
            graph.nodes(data=True),
            key=lambda item: (-item[1].get("count", 0), item[1].get("label", item[0])),
        ):
            writer.writerow(
                {
                    "label": data.get("label", node),
                    "node_id": node,
                    "node_type": data.get("node_type", ""),
                    "count": data.get("count", 0),
                    "degree": data.get("degree", 0),
                    "weighted_degree": data.get("weighted_degree", 0.0),
                    "degree_centrality": data.get("degree_centrality", 0.0),
                    "betweenness_centrality": data.get("betweenness_centrality", 0.0),
                    "community": data.get("community", -1),
                }
            )


def write_edges(graph: nx.Graph, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source", "source_label", "source_type", "target", "target_label", "target_type", "weight"]
    with output_file.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for source, target, data in sorted(
            graph.edges(data=True), key=lambda item: (-item[2].get("weight", 0), item[0], item[1])
        ):
            source_data = graph.nodes[source]
            target_data = graph.nodes[target]
            writer.writerow(
                {
                    "source": source,
                    "source_label": source_data.get("label", source),
                    "source_type": source_data.get("node_type", ""),
                    "target": target,
                    "target_label": target_data.get("label", target),
                    "target_type": target_data.get("node_type", ""),
                    "weight": data.get("weight", 1),
                }
            )


def write_summary(
    graph: nx.Graph,
    output_file: Path,
    *,
    input_file: Path,
    input_rows: int,
    ok_rows: int,
    used_rows: int,
    min_confidence: float,
    min_sentiment_confidence: float,
    min_topic_probability: float,
    min_edge_weight: int,
    include_fallback: bool,
    include_sentiment_fallback: bool,
    include_neutral_polarity: bool,
    include_topic_noise: bool,
    limit: int | None,
    graphs: dict[str, tuple[nx.Graph, int]],
    optional_inputs: dict[str, Any],
) -> None:
    summary = {
        "input_file": str(input_file),
        "input_rows": input_rows,
        "ok_rows": ok_rows,
        "used_rows": used_rows,
        "skipped_rows": input_rows - used_rows,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "top_nodes": summarize_top_nodes(graph),
        "top_edges": summarize_top_edges(graph),
        "networks": {
            name: summarize_network(network_graph, used_rows=network_used_rows)
            for name, (network_graph, network_used_rows) in sorted(graphs.items())
        },
        "optional_inputs": optional_inputs,
        "parameters": {
            "limit": limit,
            "min_confidence": min_confidence,
            "min_sentiment_confidence": min_sentiment_confidence,
            "min_topic_probability": min_topic_probability,
            "min_edge_weight": min_edge_weight,
            "include_fallback": include_fallback,
            "include_sentiment_fallback": include_sentiment_fallback,
            "include_neutral_polarity": include_neutral_polarity,
            "include_topic_noise": include_topic_noise,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_network(graph: nx.Graph, *, used_rows: int) -> dict[str, Any]:
    return {
        "used_rows": used_rows,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "top_nodes": summarize_top_nodes(graph),
        "top_edges": summarize_top_edges(graph),
    }


def summarize_top_edges(graph: nx.Graph) -> list[dict[str, Any]]:
    return [
        {"source": source, "target": target, "weight": data.get("weight", 1)}
        for source, target, data in sorted(
            graph.edges(data=True), key=lambda item: (-item[2].get("weight", 0), item[0], item[1])
        )[:10]
    ]


def summarize_top_nodes(graph: nx.Graph) -> list[dict[str, Any]]:
    return [
        {
            "label": data.get("label", node),
            "node_id": node,
            "node_type": data.get("node_type", ""),
            "count": data.get("count", 0),
            "weighted_degree": data.get("weighted_degree", 0.0),
        }
        for node, data in sorted(
            graph.nodes(data=True),
            key=lambda item: (-item[1].get("count", 0), item[1].get("label", item[0])),
        )[:10]
    ]


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
    parser = argparse.ArgumentParser(description="Build NetworkX co-occurrence graphs from analysis labels.")
    parser.add_argument(
        "--input-file",
        "--content-file",
        dest="input_file",
        default=str(DEFAULT_INPUT_FILE),
        help="Content-analysis JSONL file. Defaults to data/content_analysis/comment_labels.jsonl.",
    )
    parser.add_argument(
        "--sentiment-file",
        default=str(DEFAULT_SENTIMENT_FILE),
        help="Sentiment-analysis JSONL file. Skipped when the file does not exist.",
    )
    parser.add_argument(
        "--topic-file",
        default=str(DEFAULT_TOPIC_FILE),
        help="BERTopic comment_topics.csv file. Skipped when the file does not exist.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder for co-occurrence outputs. Defaults to data/cooccurrence_analysis.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only consider the first N content rows.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only include content labels with at least this confidence.",
    )
    parser.add_argument(
        "--min-sentiment-confidence",
        type=float,
        default=None,
        help="Only include sentiment labels with at least this confidence. Defaults to --min-confidence.",
    )
    parser.add_argument(
        "--min-topic-probability",
        type=float,
        default=0.0,
        help="Only include topic assignments with at least this probability.",
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
        help="Include fallback content labels such as 无法归类/无关讨论.",
    )
    parser.add_argument(
        "--include-sentiment-fallback",
        action="store_true",
        help="Include fallback sentiment labels such as 中性/无法判断.",
    )
    parser.add_argument(
        "--exclude-neutral-polarity",
        action="store_true",
        help="Exclude neutral sentiment polarity from polarity networks.",
    )
    parser.add_argument(
        "--include-topic-noise",
        action="store_true",
        help="Include BERTopic noise topic -1 in topic networks.",
    )
    parser.add_argument(
        "--no-sentiment",
        action="store_true",
        help="Skip sentiment-derived networks even if the sentiment file exists.",
    )
    parser.add_argument(
        "--no-topic",
        action="store_true",
        help="Skip topic-derived networks even if the topic file exists.",
    )
    args = parser.parse_args()

    report = analyze_cooccurrence(
        input_file=args.input_file,
        output_dir=args.output_dir,
        sentiment_file=None if args.no_sentiment else args.sentiment_file,
        topic_file=None if args.no_topic else args.topic_file,
        limit=args.limit,
        min_confidence=args.min_confidence,
        min_sentiment_confidence=args.min_sentiment_confidence,
        min_topic_probability=args.min_topic_probability,
        min_edge_weight=args.min_edge_weight,
        include_fallback=args.include_fallback,
        include_sentiment_fallback=args.include_sentiment_fallback,
        include_neutral_polarity=not args.exclude_neutral_polarity,
        include_topic_noise=args.include_topic_noise,
    )
    print(f"Input rows: {report.input_rows}")
    print(f"OK rows: {report.ok_rows}")
    print(f"Used rows: {report.used_rows}")
    print(f"Skipped rows: {report.skipped_rows}")
    print(f"Content nodes: {report.node_count}")
    print(f"Content edges: {report.edge_count}")
    for name, network_report in sorted(report.network_reports.items()):
        print(f"{name}: rows={network_report.used_rows}, nodes={network_report.node_count}, edges={network_report.edge_count}")
    print(f"Summary file: {report.outputs.summary_file}")


if __name__ == "__main__":
    main()
