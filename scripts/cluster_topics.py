from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloader import CommentRecord, DataLoader


DEFAULT_INPUT_FILE = Path("data/cleaned/comments_cleaned.csv")
DEFAULT_OUTPUT_DIR = Path("data/topic_clustering")
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MIN_TEXT_LENGTH = 4
DEFAULT_MIN_TOPIC_SIZE = 15

COMMENT_TOPICS_FILE_NAME = "comment_topics.csv"
TOPIC_INFO_FILE_NAME = "topic_info.csv"
REPRESENTATIVE_DOCS_FILE_NAME = "topic_representative_docs.jsonl"
METADATA_FILE_NAME = "run_metadata.json"
MODEL_DIR_NAME = "model"


class TopicModel(Protocol):
    def fit_transform(self, documents: list[str]) -> tuple[list[int], object]:
        ...

    def get_topic_info(self) -> object:
        ...

    def get_topic(self, topic: int) -> object:
        ...

    def get_representative_docs(self, topic: int | None = None) -> object:
        ...

    def save(self, path: str | Path) -> None:
        ...


@dataclass(frozen=True)
class TopicDocument:
    record_index: int
    record: CommentRecord
    text: str


@dataclass(frozen=True)
class TopicClusteringOutputs:
    comment_topics_file: Path
    topic_info_file: Path
    representative_docs_file: Path
    metadata_file: Path
    model_dir: Path | None


@dataclass(frozen=True)
class TopicClusteringReport:
    loaded_records: int
    considered_records: int
    clustered_records: int
    skipped_short_records: int
    total_topics: int
    noise_records: int
    outputs: TopicClusteringOutputs


def cluster_topics(
    input_file: str | Path = DEFAULT_INPUT_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    limit: int | None = None,
    min_text_length: int = DEFAULT_MIN_TEXT_LENGTH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    min_topic_size: int = DEFAULT_MIN_TOPIC_SIZE,
    save_model: bool = False,
    topic_model: TopicModel | None = None,
) -> TopicClusteringReport:
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to 0")
    if min_text_length < 1:
        raise ValueError("min_text_length must be greater than or equal to 1")
    if min_topic_size < 2:
        raise ValueError("min_topic_size must be greater than or equal to 2")

    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = DataLoader().load(input_path)
    considered_records = records[:limit] if limit is not None else records
    topic_documents, skipped_short_records = select_topic_documents(considered_records, min_text_length)

    outputs = TopicClusteringOutputs(
        comment_topics_file=output_path / COMMENT_TOPICS_FILE_NAME,
        topic_info_file=output_path / TOPIC_INFO_FILE_NAME,
        representative_docs_file=output_path / REPRESENTATIVE_DOCS_FILE_NAME,
        metadata_file=output_path / METADATA_FILE_NAME,
        model_dir=(output_path / MODEL_DIR_NAME) if save_model else None,
    )

    if not topic_documents:
        write_comment_topics([], [], None, outputs.comment_topics_file)
        write_topic_info([], None, outputs.topic_info_file)
        write_representative_docs({}, outputs.representative_docs_file)
        write_metadata(
            outputs.metadata_file,
            input_path=input_path,
            loaded_records=len(records),
            considered_records=len(considered_records),
            clustered_records=0,
            skipped_short_records=skipped_short_records,
            total_topics=0,
            noise_records=0,
            limit=limit,
            min_text_length=min_text_length,
            embedding_model=embedding_model,
            min_topic_size=min_topic_size,
            save_model=save_model,
            model_dir=outputs.model_dir,
        )
        return TopicClusteringReport(
            loaded_records=len(records),
            considered_records=len(considered_records),
            clustered_records=0,
            skipped_short_records=skipped_short_records,
            total_topics=0,
            noise_records=0,
            outputs=outputs,
        )

    model = topic_model or create_topic_model(embedding_model=embedding_model, min_topic_size=min_topic_size)
    documents = [document.text for document in topic_documents]
    topics, probabilities = model.fit_transform(documents)

    write_comment_topics(topic_documents, topics, probabilities, outputs.comment_topics_file)
    topic_info_rows = normalize_topic_info(model)
    write_topic_info(topic_info_rows, model, outputs.topic_info_file)
    representative_docs = normalize_representative_docs(model, topic_info_rows)
    write_representative_docs(representative_docs, outputs.representative_docs_file)

    if save_model and outputs.model_dir is not None:
        outputs.model_dir.parent.mkdir(parents=True, exist_ok=True)
        model.save(outputs.model_dir)

    total_topics = len({topic for topic in topics if topic != -1})
    noise_records = sum(1 for topic in topics if topic == -1)
    write_metadata(
        outputs.metadata_file,
        input_path=input_path,
        loaded_records=len(records),
        considered_records=len(considered_records),
        clustered_records=len(topic_documents),
        skipped_short_records=skipped_short_records,
        total_topics=total_topics,
        noise_records=noise_records,
        limit=limit,
        min_text_length=min_text_length,
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        save_model=save_model,
        model_dir=outputs.model_dir,
    )

    return TopicClusteringReport(
        loaded_records=len(records),
        considered_records=len(considered_records),
        clustered_records=len(topic_documents),
        skipped_short_records=skipped_short_records,
        total_topics=total_topics,
        noise_records=noise_records,
        outputs=outputs,
    )


def create_topic_model(*, embedding_model: str, min_topic_size: int) -> TopicModel:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    sentence_model = SentenceTransformer(embedding_model)
    vectorizer_model = CountVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    return BERTopic(
        embedding_model=sentence_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=min_topic_size,
        calculate_probabilities=True,
        verbose=True,
    )


def select_topic_documents(records: list[CommentRecord], min_text_length: int) -> tuple[list[TopicDocument], int]:
    topic_documents: list[TopicDocument] = []
    skipped_short_records = 0
    for record_index, record in enumerate(records):
        text = record.content.strip()
        if len(text) < min_text_length:
            skipped_short_records += 1
            continue
        topic_documents.append(TopicDocument(record_index=record_index, record=record, text=text))

    return topic_documents, skipped_short_records


def write_comment_topics(
    topic_documents: list[TopicDocument],
    topics: Iterable[int],
    probabilities: object,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_index",
        "topic",
        "topic_probability",
        "username",
        "gender",
        "content",
        "comment_time",
        "likes",
        "ip_location",
        "signature",
        "feature",
    ]
    with output_file.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for position, (topic_document, topic) in enumerate(zip(topic_documents, topics)):
            writer.writerow(
                {
                    "record_index": topic_document.record_index,
                    "topic": topic,
                    "topic_probability": extract_probability(probabilities, position, topic),
                    **asdict(topic_document.record),
                }
            )


def extract_probability(probabilities: object, position: int, topic: int) -> float | str:
    if probabilities is None:
        return ""

    try:
        value = probabilities[position]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return ""

    if isinstance(value, int | float):
        return float(value)

    try:
        values = list(value)
    except TypeError:
        return ""

    if not values:
        return ""
    if topic >= 0 and topic < len(values) and isinstance(values[topic], int | float):
        return float(values[topic])

    numeric_values = [float(item) for item in values if isinstance(item, int | float)]
    return max(numeric_values) if numeric_values else ""


def normalize_topic_info(topic_model: TopicModel) -> list[dict[str, Any]]:
    topic_info = topic_model.get_topic_info()
    if hasattr(topic_info, "to_dict"):
        return [dict(row) for row in topic_info.to_dict("records")]
    return [dict(row) for row in topic_info]


def write_topic_info(topic_info_rows: list[dict[str, Any]], topic_model: TopicModel | None, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = [normalize_topic_info_row(row, topic_model) for row in topic_info_rows]
    fieldnames = ordered_topic_info_fieldnames(rows)
    with output_file.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_topic_info_row(row: dict[str, Any], topic_model: TopicModel | None) -> dict[str, Any]:
    normalized = {key: serialize_cell(value) for key, value in row.items()}
    topic_id = get_topic_id(row)
    if "Keywords" not in normalized and topic_id is not None and topic_model is not None:
        normalized["Keywords"] = serialize_topic_keywords(topic_model.get_topic(topic_id))
    return normalized


def ordered_topic_info_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = ["Topic", "Count", "Name", "Keywords", "Representation"]
    seen = {field for row in rows for field in row}
    return [field for field in preferred if field in seen] + sorted(seen - set(preferred))


def get_topic_id(row: dict[str, Any]) -> int | None:
    value = row.get("Topic", row.get("topic"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def serialize_topic_keywords(topic_words: object) -> str:
    if topic_words is None:
        return ""
    try:
        return ";".join(f"{word}:{score:.4f}" for word, score in topic_words)  # type: ignore[misc]
    except (TypeError, ValueError):
        return serialize_cell(topic_words)


def serialize_cell(value: object) -> str:
    if isinstance(value, list | tuple):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def normalize_representative_docs(topic_model: TopicModel, topic_info_rows: list[dict[str, Any]]) -> dict[int, list[str]]:
    try:
        representative_docs = topic_model.get_representative_docs()
    except TypeError:
        representative_docs = None

    if isinstance(representative_docs, dict):
        return {int(topic): list(docs) for topic, docs in representative_docs.items()}

    docs_by_topic: dict[int, list[str]] = {}
    for row in topic_info_rows:
        topic_id = get_topic_id(row)
        if topic_id is None:
            continue
        try:
            docs = topic_model.get_representative_docs(topic_id)
        except TypeError:
            docs = []
        docs_by_topic[topic_id] = list(docs or [])

    return docs_by_topic


def write_representative_docs(representative_docs: dict[int, list[str]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="\n") as jsonl_file:
        for topic in sorted(representative_docs):
            jsonl_file.write(
                json.dumps(
                    {
                        "topic": topic,
                        "representative_docs": representative_docs[topic],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_metadata(
    output_file: Path,
    *,
    input_path: Path,
    loaded_records: int,
    considered_records: int,
    clustered_records: int,
    skipped_short_records: int,
    total_topics: int,
    noise_records: int,
    limit: int | None,
    min_text_length: int,
    embedding_model: str,
    min_topic_size: int,
    save_model: bool,
    model_dir: Path | None,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input_file": str(input_path),
        "loaded_records": loaded_records,
        "considered_records": considered_records,
        "clustered_records": clustered_records,
        "skipped_short_records": skipped_short_records,
        "total_topics": total_topics,
        "noise_records": noise_records,
        "parameters": {
            "limit": limit,
            "min_text_length": min_text_length,
            "embedding_model": embedding_model,
            "min_topic_size": min_topic_size,
            "save_model": save_model,
            "model_dir": str(model_dir) if model_dir is not None else None,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover comment topics with BERTopic.")
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT_FILE),
        help="CSV file to cluster. Defaults to data/cleaned/comments_cleaned.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder for clustering outputs. Defaults to data/topic_clustering.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only consider the first N input records.")
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=DEFAULT_MIN_TEXT_LENGTH,
        help="Skip comments shorter than this length after stripping whitespace.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name or path.",
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=DEFAULT_MIN_TOPIC_SIZE,
        help="Minimum HDBSCAN topic size.",
    )
    parser.add_argument("--save-model", action="store_true", help="Save the BERTopic model under the output folder.")
    args = parser.parse_args()

    report = cluster_topics(
        input_file=args.input_file,
        output_dir=args.output_dir,
        limit=args.limit,
        min_text_length=args.min_text_length,
        embedding_model=args.embedding_model,
        min_topic_size=args.min_topic_size,
        save_model=args.save_model,
    )
    print(f"Loaded records: {report.loaded_records}")
    print(f"Considered records: {report.considered_records}")
    print(f"Clustered records: {report.clustered_records}")
    print(f"Skipped short records: {report.skipped_short_records}")
    print(f"Total topics: {report.total_topics}")
    print(f"Noise records: {report.noise_records}")
    print(f"Comment topics file: {report.outputs.comment_topics_file}")
    print(f"Topic info file: {report.outputs.topic_info_file}")
    print(f"Representative docs file: {report.outputs.representative_docs_file}")
    print(f"Metadata file: {report.outputs.metadata_file}")
    if report.outputs.model_dir is not None:
        print(f"Model dir: {report.outputs.model_dir}")


if __name__ == "__main__":
    main()
