from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_full_analysis import run_full_analysis
from scripts.run_pipeline import PipelineOutputs


def test_run_full_analysis_orchestrates_all_stages(monkeypatch, tmp_path):
    calls = []
    raw_dir = tmp_path / "raw"
    cleaned_file = tmp_path / "cleaned" / "comments_cleaned.csv"
    deduplicated_file = tmp_path / "deduplicated" / "comments_deduplicated.csv"
    flagged_file = tmp_path / "cleaned" / "rejected_or_flagged_comments.csv"
    config_file = tmp_path / "config.json"
    content_dir = tmp_path / "content"
    sentiment_dir = tmp_path / "sentiment"
    topic_dir = tmp_path / "topic"
    cooccurrence_dir = tmp_path / "cooccurrence"
    report_dir = tmp_path / "report"
    state_file = tmp_path / "state.json"

    def fake_run_pipeline(**kwargs):
        calls.append(("pipeline", kwargs))
        write_file(deduplicated_file)
        write_file(cleaned_file)
        write_file(flagged_file)
        return PipelineOutputs(
            deduplicated_file=deduplicated_file,
            cleaned_file=cleaned_file,
            flagged_file=flagged_file,
        )

    def fake_analyze_content(**kwargs):
        calls.append(("content", kwargs))
        write_file(content_dir / "comment_labels.jsonl")
        return SimpleNamespace(output_file=content_dir / "comment_labels.jsonl")

    def fake_analyze_sentiment(**kwargs):
        calls.append(("sentiment", kwargs))
        write_file(sentiment_dir / "comment_sentiment.jsonl")
        return SimpleNamespace(output_file=sentiment_dir / "comment_sentiment.jsonl")

    def fake_cluster_topics(**kwargs):
        calls.append(("topic", kwargs))
        write_file(topic_dir / "comment_topics.csv")
        write_file(topic_dir / "topic_info.csv")
        write_file(topic_dir / "topic_representative_docs.jsonl")
        write_file(topic_dir / "run_metadata.json")
        return SimpleNamespace(
            outputs=SimpleNamespace(
                comment_topics_file=topic_dir / "comment_topics.csv",
                topic_info_file=topic_dir / "topic_info.csv",
                representative_docs_file=topic_dir / "topic_representative_docs.jsonl",
                metadata_file=topic_dir / "run_metadata.json",
                model_dir=None,
            )
        )

    def fake_analyze_cooccurrence(**kwargs):
        calls.append(("cooccurrence", kwargs))
        write_file(cooccurrence_dir / "summary.json")
        return SimpleNamespace(outputs=SimpleNamespace(summary_file=cooccurrence_dir / "summary.json"))

    def fake_build_analysis_report(**kwargs):
        calls.append(("report", kwargs))
        write_file(report_dir / "report.html")
        write_file(report_dir / "interactive_report.html")
        write_file(report_dir / "report_metadata.json")
        return SimpleNamespace(
            static_html=report_dir / "report.html",
            interactive_html=report_dir / "interactive_report.html",
            metadata_file=report_dir / "report_metadata.json",
        )

    monkeypatch.setattr("scripts.run_full_analysis.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("scripts.run_full_analysis.analyze_content", fake_analyze_content)
    monkeypatch.setattr("scripts.run_full_analysis.analyze_sentiment", fake_analyze_sentiment)
    monkeypatch.setattr("scripts.run_full_analysis.cluster_topics", fake_cluster_topics)
    monkeypatch.setattr("scripts.run_full_analysis.analyze_cooccurrence", fake_analyze_cooccurrence)
    monkeypatch.setattr("scripts.run_full_analysis.build_analysis_report", fake_build_analysis_report)

    outputs = run_full_analysis(
        raw_data_dir=raw_dir,
        config_file=config_file,
        content_output_dir=content_dir,
        sentiment_output_dir=sentiment_dir,
        topic_output_dir=topic_dir,
        cooccurrence_output_dir=cooccurrence_dir,
        report_output_dir=report_dir,
        state_file=state_file,
        limit=5,
        overwrite_llm=True,
        min_confidence=0.7,
        min_sentiment_confidence=0.8,
        min_edge_weight=2,
        include_fallback=True,
        include_sentiment_fallback=True,
        include_neutral_polarity=False,
        include_topic_noise=True,
        min_text_length=3,
        embedding_model="fake-embedding",
        min_topic_size=4,
        save_topic_model=True,
        top_n=8,
        stopwords_file=tmp_path / "stopwords.txt",
    )

    assert [name for name, _ in calls] == ["pipeline", "content", "sentiment", "topic", "cooccurrence", "report"]
    assert outputs.deduplicated_file == deduplicated_file
    assert outputs.cleaned_file == cleaned_file
    assert outputs.flagged_file == flagged_file
    assert outputs.content_file == content_dir / "comment_labels.jsonl"
    assert outputs.sentiment_file == sentiment_dir / "comment_sentiment.jsonl"
    assert outputs.topic_file == topic_dir / "comment_topics.csv"
    assert outputs.topic_info_file == topic_dir / "topic_info.csv"
    assert outputs.cooccurrence_summary_file == cooccurrence_dir / "summary.json"
    assert outputs.report_html == report_dir / "report.html"
    assert outputs.interactive_report_html == report_dir / "interactive_report.html"
    assert outputs.report_metadata_file == report_dir / "report_metadata.json"

    content_kwargs = calls[1][1]
    assert content_kwargs["input_file"] == cleaned_file
    assert content_kwargs["output_dir"] == content_dir
    assert content_kwargs["config_file"] == config_file
    assert content_kwargs["limit"] == 5
    assert content_kwargs["overwrite"] is True

    topic_kwargs = calls[3][1]
    assert topic_kwargs["input_file"] == cleaned_file
    assert topic_kwargs["output_dir"] == topic_dir
    assert topic_kwargs["limit"] == 5
    assert topic_kwargs["min_text_length"] == 3
    assert topic_kwargs["embedding_model"] == "fake-embedding"
    assert topic_kwargs["min_topic_size"] == 4
    assert topic_kwargs["save_model"] is True

    cooccurrence_kwargs = calls[4][1]
    assert cooccurrence_kwargs["input_file"] == content_dir / "comment_labels.jsonl"
    assert cooccurrence_kwargs["cleaned_file"] == cleaned_file
    assert cooccurrence_kwargs["sentiment_file"] == sentiment_dir / "comment_sentiment.jsonl"
    assert cooccurrence_kwargs["topic_file"] == topic_dir / "comment_topics.csv"
    assert cooccurrence_kwargs["min_confidence"] == 0.7
    assert cooccurrence_kwargs["min_sentiment_confidence"] == 0.8
    assert cooccurrence_kwargs["min_edge_weight"] == 2
    assert cooccurrence_kwargs["include_fallback"] is True
    assert cooccurrence_kwargs["include_sentiment_fallback"] is True
    assert cooccurrence_kwargs["include_neutral_polarity"] is False
    assert cooccurrence_kwargs["include_topic_noise"] is True

    report_kwargs = calls[5][1]
    assert report_kwargs["cleaned_file"] == cleaned_file
    assert report_kwargs["content_file"] == content_dir / "comment_labels.jsonl"
    assert report_kwargs["sentiment_file"] == sentiment_dir / "comment_sentiment.jsonl"
    assert report_kwargs["topic_file"] == topic_dir / "comment_topics.csv"
    assert report_kwargs["topic_info_file"] == topic_dir / "topic_info.csv"
    assert report_kwargs["cooccurrence_dir"] == cooccurrence_dir
    assert report_kwargs["output_dir"] == report_dir
    assert report_kwargs["top_n"] == 8


def test_run_full_analysis_skip_flags_use_expected_outputs(monkeypatch, tmp_path):
    cleaned_file = tmp_path / "cleaned" / "comments_cleaned.csv"

    monkeypatch.setattr(
        "scripts.run_full_analysis.run_pipeline",
        lambda **kwargs: pipeline_outputs_with_files(
            deduplicated_file=tmp_path / "deduplicated" / "comments_deduplicated.csv",
            cleaned_file=cleaned_file,
            flagged_file=tmp_path / "cleaned" / "rejected_or_flagged_comments.csv",
        ),
    )
    monkeypatch.setattr("scripts.run_full_analysis.analyze_content", fail_if_called)
    monkeypatch.setattr("scripts.run_full_analysis.analyze_sentiment", fail_if_called)
    monkeypatch.setattr("scripts.run_full_analysis.cluster_topics", fail_if_called)
    monkeypatch.setattr("scripts.run_full_analysis.analyze_cooccurrence", fail_if_called)
    monkeypatch.setattr("scripts.run_full_analysis.build_analysis_report", fail_if_called)

    outputs = run_full_analysis(
        raw_data_dir=tmp_path / "raw",
        content_output_dir=tmp_path / "content",
        sentiment_output_dir=tmp_path / "sentiment",
        topic_output_dir=tmp_path / "topic",
        cooccurrence_output_dir=tmp_path / "cooccurrence",
        report_output_dir=tmp_path / "report",
        state_file=tmp_path / "state.json",
        skip_content=True,
        skip_sentiment=True,
        skip_topic=True,
        skip_cooccurrence=True,
        skip_report=True,
    )

    assert outputs.content_file == tmp_path / "content" / "comment_labels.jsonl"
    assert outputs.sentiment_file == tmp_path / "sentiment" / "comment_sentiment.jsonl"
    assert outputs.topic_file == tmp_path / "topic" / "comment_topics.csv"
    assert outputs.topic_info_file == tmp_path / "topic" / "topic_info.csv"
    assert outputs.cooccurrence_summary_file == tmp_path / "cooccurrence" / "summary.json"
    assert outputs.report_html == tmp_path / "report" / "report.html"
    assert outputs.interactive_report_html == tmp_path / "report" / "interactive_report.html"
    assert outputs.report_metadata_file == tmp_path / "report" / "report_metadata.json"


def test_run_full_analysis_validates_limit():
    with pytest.raises(ValueError, match="limit"):
        run_full_analysis(limit=-1)


def test_run_full_analysis_resume_skips_completed_stage_outputs(monkeypatch, tmp_path):
    calls = []
    raw_dir = tmp_path / "raw"
    cleaned_file = tmp_path / "cleaned" / "comments_cleaned.csv"
    deduplicated_file = tmp_path / "deduplicated" / "comments_deduplicated.csv"
    flagged_file = tmp_path / "cleaned" / "rejected_or_flagged_comments.csv"
    content_dir = tmp_path / "content"
    sentiment_dir = tmp_path / "sentiment"
    topic_dir = tmp_path / "topic"
    cooccurrence_dir = tmp_path / "cooccurrence"
    report_dir = tmp_path / "report"
    state_file = tmp_path / "state.json"

    def fake_run_pipeline(**kwargs):
        calls.append("pipeline")
        return pipeline_outputs_with_files(
            deduplicated_file=deduplicated_file,
            cleaned_file=cleaned_file,
            flagged_file=flagged_file,
        )

    def fake_analyze_content(**kwargs):
        calls.append("content")
        write_file(content_dir / "comment_labels.jsonl")
        return SimpleNamespace(output_file=content_dir / "comment_labels.jsonl")

    def fake_analyze_sentiment(**kwargs):
        calls.append("sentiment")
        write_file(sentiment_dir / "comment_sentiment.jsonl")
        return SimpleNamespace(output_file=sentiment_dir / "comment_sentiment.jsonl")

    def fake_cluster_topics(**kwargs):
        calls.append("topic")
        write_file(topic_dir / "comment_topics.csv")
        write_file(topic_dir / "topic_info.csv")
        write_file(topic_dir / "topic_representative_docs.jsonl")
        write_file(topic_dir / "run_metadata.json")
        return SimpleNamespace(
            outputs=SimpleNamespace(
                comment_topics_file=topic_dir / "comment_topics.csv",
                topic_info_file=topic_dir / "topic_info.csv",
                representative_docs_file=topic_dir / "topic_representative_docs.jsonl",
                metadata_file=topic_dir / "run_metadata.json",
                model_dir=None,
            )
        )

    def fake_analyze_cooccurrence(**kwargs):
        calls.append("cooccurrence")
        write_file(cooccurrence_dir / "summary.json")
        return SimpleNamespace(outputs=SimpleNamespace(summary_file=cooccurrence_dir / "summary.json"))

    def fake_build_analysis_report(**kwargs):
        calls.append("report")
        write_file(report_dir / "report.html")
        write_file(report_dir / "interactive_report.html")
        write_file(report_dir / "report_metadata.json")
        return SimpleNamespace(
            static_html=report_dir / "report.html",
            interactive_html=report_dir / "interactive_report.html",
            metadata_file=report_dir / "report_metadata.json",
        )

    monkeypatch.setattr("scripts.run_full_analysis.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("scripts.run_full_analysis.analyze_content", fake_analyze_content)
    monkeypatch.setattr("scripts.run_full_analysis.analyze_sentiment", fake_analyze_sentiment)
    monkeypatch.setattr("scripts.run_full_analysis.cluster_topics", fake_cluster_topics)
    monkeypatch.setattr("scripts.run_full_analysis.analyze_cooccurrence", fake_analyze_cooccurrence)
    monkeypatch.setattr("scripts.run_full_analysis.build_analysis_report", fake_build_analysis_report)

    kwargs = {
        "raw_data_dir": raw_dir,
        "content_output_dir": content_dir,
        "sentiment_output_dir": sentiment_dir,
        "topic_output_dir": topic_dir,
        "cooccurrence_output_dir": cooccurrence_dir,
        "report_output_dir": report_dir,
        "state_file": state_file,
        "limit": 3,
    }
    run_full_analysis(**kwargs)
    run_full_analysis(**kwargs)

    assert calls == ["pipeline", "content", "sentiment", "topic", "cooccurrence", "report"]


def test_run_full_analysis_retries_llm_stage_until_error_free(monkeypatch, tmp_path):
    calls = []
    raw_dir = tmp_path / "raw"
    cleaned_file = tmp_path / "cleaned" / "comments_cleaned.csv"
    content_dir = tmp_path / "content"
    state_file = tmp_path / "state.json"

    monkeypatch.setattr(
        "scripts.run_full_analysis.run_pipeline",
        lambda **kwargs: pipeline_outputs_with_files(
            deduplicated_file=tmp_path / "deduplicated" / "comments_deduplicated.csv",
            cleaned_file=cleaned_file,
            flagged_file=tmp_path / "cleaned" / "rejected_or_flagged_comments.csv",
        ),
    )

    def fake_analyze_content(**kwargs):
        calls.append("content")
        write_file(content_dir / "comment_labels.jsonl")
        return SimpleNamespace(output_file=content_dir / "comment_labels.jsonl", error_records=1)

    monkeypatch.setattr("scripts.run_full_analysis.analyze_content", fake_analyze_content)
    monkeypatch.setattr("scripts.run_full_analysis.analyze_sentiment", fail_if_called)
    monkeypatch.setattr("scripts.run_full_analysis.cluster_topics", fail_if_called)
    monkeypatch.setattr("scripts.run_full_analysis.analyze_cooccurrence", fail_if_called)
    monkeypatch.setattr("scripts.run_full_analysis.build_analysis_report", fail_if_called)

    kwargs = {
        "raw_data_dir": raw_dir,
        "content_output_dir": content_dir,
        "state_file": state_file,
        "skip_sentiment": True,
        "skip_topic": True,
        "skip_cooccurrence": True,
        "skip_report": True,
    }
    run_full_analysis(**kwargs)
    run_full_analysis(**kwargs)

    assert calls == ["content", "content"]


def pipeline_outputs_with_files(*, deduplicated_file: Path, cleaned_file: Path, flagged_file: Path) -> PipelineOutputs:
    write_file(deduplicated_file)
    write_file(cleaned_file)
    write_file(flagged_file)
    return PipelineOutputs(
        deduplicated_file=deduplicated_file,
        cleaned_file=cleaned_file,
        flagged_file=flagged_file,
    )


def write_file(path: Path, content: str = "ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def fail_if_called(**kwargs):
    raise AssertionError("This stage should have been skipped")
