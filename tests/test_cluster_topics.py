import csv
import json
import sys
import types
from pathlib import Path

import pytest

from scripts.cluster_topics import create_topic_model, cluster_topics


class FakeTopicModel:
    def __init__(self, topics=None, probabilities=None):
        self.topics = topics or [0, 1, 0]
        self.probabilities = probabilities or [0.91, 0.77, 0.83]
        self.fit_documents = None
        self.saved_path = None

    def fit_transform(self, documents):
        self.fit_documents = documents
        return self.topics[: len(documents)], self.probabilities[: len(documents)]

    def get_topic_info(self):
        return [
            {"Topic": 0, "Count": 2, "Name": "0_ai_tools", "Representation": ["AI", "工具"]},
            {"Topic": 1, "Count": 1, "Name": "1_copyright", "Representation": ["版权", "训练"]},
        ]

    def get_topic(self, topic):
        topics = {
            0: [("AI", 0.25), ("工具", 0.2)],
            1: [("版权", 0.3), ("训练", 0.18)],
        }
        return topics.get(topic, [])

    def get_representative_docs(self, topic=None):
        docs = {
            0: ["AI 工具真的好用", "AI 可以提高效率"],
            1: ["版权训练问题很大"],
        }
        if topic is None:
            return docs
        return docs.get(topic, [])

    def save(self, path):
        self.saved_path = Path(path)


def test_cluster_topics_respects_limit_filters_short_text_and_writes_outputs(tmp_path):
    input_file = tmp_path / "comments_cleaned.csv"
    output_dir = tmp_path / "topic_clustering"
    write_comment_csv(
        input_file,
        [
            "短",
            "AI 工具真的好用",
            "版权训练问题很大",
            "这一条不应该被 limit 处理",
        ],
    )
    topic_model = FakeTopicModel(topics=[0, 1], probabilities=[0.92, 0.81])

    report = cluster_topics(
        input_file=input_file,
        output_dir=output_dir,
        limit=3,
        min_text_length=4,
        min_topic_size=2,
        topic_model=topic_model,
    )

    comment_rows = read_csv(output_dir / "comment_topics.csv")
    topic_rows = read_csv(output_dir / "topic_info.csv")
    representative_rows = read_jsonl(output_dir / "topic_representative_docs.jsonl")
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert topic_model.fit_documents == ["AI 工具真的好用", "版权训练问题很大"]
    assert report.loaded_records == 4
    assert report.considered_records == 3
    assert report.clustered_records == 2
    assert report.skipped_short_records == 1
    assert report.total_topics == 2
    assert report.noise_records == 0
    assert [row["record_index"] for row in comment_rows] == ["1", "2"]
    assert [row["topic"] for row in comment_rows] == ["0", "1"]
    assert [row["topic_probability"] for row in comment_rows] == ["0.92", "0.81"]
    assert comment_rows[0]["content"] == "AI 工具真的好用"
    assert topic_rows[0]["Keywords"] == "AI:0.2500;工具:0.2000"
    assert representative_rows == [
        {"topic": 0, "representative_docs": ["AI 工具真的好用", "AI 可以提高效率"]},
        {"topic": 1, "representative_docs": ["版权训练问题很大"]},
    ]
    assert metadata["loaded_records"] == 4
    assert metadata["considered_records"] == 3
    assert metadata["clustered_records"] == 2
    assert metadata["skipped_short_records"] == 1
    assert metadata["total_topics"] == 2
    assert metadata["noise_records"] == 0
    assert metadata["parameters"]["limit"] == 3
    assert metadata["parameters"]["min_text_length"] == 4
    assert metadata["parameters"]["min_topic_size"] == 2


def test_cluster_topics_counts_noise_and_saves_model(tmp_path):
    input_file = tmp_path / "comments_cleaned.csv"
    output_dir = tmp_path / "topic_clustering"
    write_comment_csv(input_file, ["AI 工具真的好用", "噪声评论内容", "版权训练问题很大"])
    topic_model = FakeTopicModel(topics=[0, -1, 1], probabilities=[0.92, 0.1, 0.81])

    report = cluster_topics(
        input_file=input_file,
        output_dir=output_dir,
        min_text_length=4,
        min_topic_size=2,
        save_model=True,
        topic_model=topic_model,
    )

    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert report.total_topics == 2
    assert report.noise_records == 1
    assert topic_model.saved_path == output_dir / "model"
    assert metadata["noise_records"] == 1
    assert metadata["parameters"]["save_model"] is True
    assert metadata["parameters"]["model_dir"] == str(output_dir / "model")


def test_cluster_topics_validates_parameters(tmp_path):
    input_file = tmp_path / "comments_cleaned.csv"
    write_comment_csv(input_file, ["AI 工具真的好用"])

    with pytest.raises(ValueError, match="limit"):
        cluster_topics(input_file=input_file, output_dir=tmp_path / "out", limit=-1, topic_model=FakeTopicModel())

    with pytest.raises(ValueError, match="min_text_length"):
        cluster_topics(input_file=input_file, output_dir=tmp_path / "out", min_text_length=0, topic_model=FakeTopicModel())

    with pytest.raises(ValueError, match="min_topic_size"):
        cluster_topics(input_file=input_file, output_dir=tmp_path / "out", min_topic_size=1, topic_model=FakeTopicModel())


def test_create_topic_model_loads_sentence_transformer_from_offline_cache(monkeypatch):
    calls = {}
    install_fake_topic_dependencies(monkeypatch, calls)

    create_topic_model(embedding_model="cached-model", min_topic_size=2)

    assert calls["sentence_transformer"] == {
        "model_name_or_path": "cached-model",
        "local_files_only": True,
    }


def test_create_topic_model_reports_missing_offline_embedding_cache(monkeypatch):
    calls = {}
    install_fake_topic_dependencies(monkeypatch, calls, sentence_transformer_error=OSError("not cached"))

    with pytest.raises(RuntimeError, match="cached-model.*offline cache"):
        create_topic_model(embedding_model="cached-model", min_topic_size=2)


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


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def install_fake_topic_dependencies(monkeypatch, calls, sentence_transformer_error=None):
    class FakeBERTopic:
        def __init__(self, **kwargs):
            calls["bertopic"] = kwargs

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            calls["hdbscan"] = kwargs

    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path, *, local_files_only=False, **kwargs):
            calls["sentence_transformer"] = {
                "model_name_or_path": model_name_or_path,
                "local_files_only": local_files_only,
            }
            if sentence_transformer_error is not None:
                raise sentence_transformer_error

    class FakeCountVectorizer:
        def __init__(self, **kwargs):
            calls["vectorizer"] = kwargs

    class FakeUMAP:
        def __init__(self, **kwargs):
            calls["umap"] = kwargs

    fake_modules = {
        "bertopic": types.SimpleNamespace(BERTopic=FakeBERTopic),
        "hdbscan": types.SimpleNamespace(HDBSCAN=FakeHDBSCAN),
        "sentence_transformers": types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
        "sklearn": types.SimpleNamespace(),
        "sklearn.feature_extraction": types.SimpleNamespace(),
        "sklearn.feature_extraction.text": types.SimpleNamespace(CountVectorizer=FakeCountVectorizer),
        "umap": types.SimpleNamespace(UMAP=FakeUMAP),
    }
    for module_name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)
