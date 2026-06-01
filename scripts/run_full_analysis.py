from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_content import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_OUTPUT_FILE_NAME as CONTENT_OUTPUT_FILE_NAME,
    DEFAULT_OUTPUT_DIR as DEFAULT_CONTENT_OUTPUT_DIR,
    analyze_content,
)
from scripts.analyze_cooccurrence import DEFAULT_OUTPUT_DIR as DEFAULT_COOCCURRENCE_OUTPUT_DIR
from scripts.analyze_cooccurrence import analyze_cooccurrence
from scripts.analyze_sentiment import DEFAULT_OUTPUT_FILE_NAME as SENTIMENT_OUTPUT_FILE_NAME
from scripts.analyze_sentiment import DEFAULT_OUTPUT_DIR as DEFAULT_SENTIMENT_OUTPUT_DIR
from scripts.analyze_sentiment import analyze_sentiment
from scripts.build_analysis_report import DEFAULT_OUTPUT_DIR as DEFAULT_REPORT_OUTPUT_DIR
from scripts.build_analysis_report import build_analysis_report
from scripts.clean_data import DEFAULT_FLAGGED_FILE_NAME, DEFAULT_OUTPUT_FILE_NAME as CLEANED_OUTPUT_FILE_NAME
from scripts.cluster_topics import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MIN_TEXT_LENGTH,
    DEFAULT_MIN_TOPIC_SIZE,
    DEFAULT_OUTPUT_DIR as DEFAULT_TOPIC_OUTPUT_DIR,
    cluster_topics,
)
from scripts.deduplicate_data import DEFAULT_OUTPUT_FILE_NAME as DEDUPLICATED_OUTPUT_FILE_NAME
from scripts.run_pipeline import CLEANED_DIR_NAME, DEDUPLICATED_DIR_NAME, DEFAULT_RAW_DATA_DIR, PipelineOutputs, run_pipeline


DEFAULT_STATE_FILE = Path("data/full_analysis_state.json")
STATE_VERSION = 1


@dataclass(frozen=True)
class FullAnalysisOutputs:
    deduplicated_file: Path
    cleaned_file: Path
    flagged_file: Path
    content_file: Path
    sentiment_file: Path
    topic_file: Path
    topic_info_file: Path
    cooccurrence_summary_file: Path
    report_html: Path
    interactive_report_html: Path
    report_metadata_file: Path


def run_full_analysis(
    *,
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    deduplicated_output_dir: str | Path | None = None,
    cleaned_output_dir: str | Path | None = None,
    config_file: str | Path = DEFAULT_CONFIG_FILE,
    content_output_dir: str | Path = DEFAULT_CONTENT_OUTPUT_DIR,
    sentiment_output_dir: str | Path = DEFAULT_SENTIMENT_OUTPUT_DIR,
    topic_output_dir: str | Path = DEFAULT_TOPIC_OUTPUT_DIR,
    cooccurrence_output_dir: str | Path = DEFAULT_COOCCURRENCE_OUTPUT_DIR,
    report_output_dir: str | Path = DEFAULT_REPORT_OUTPUT_DIR,
    state_file: str | Path = DEFAULT_STATE_FILE,
    resume: bool = True,
    limit: int | None = None,
    overwrite_llm: bool = False,
    min_confidence: float = 0.0,
    min_sentiment_confidence: float | None = None,
    min_edge_weight: int = 1,
    include_fallback: bool = False,
    include_sentiment_fallback: bool = False,
    include_neutral_polarity: bool = True,
    include_topic_noise: bool = False,
    min_text_length: int = DEFAULT_MIN_TEXT_LENGTH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    min_topic_size: int = DEFAULT_MIN_TOPIC_SIZE,
    save_topic_model: bool = False,
    top_n: int = 20,
    stopwords_file: str | Path | None = None,
    skip_content: bool = False,
    skip_sentiment: bool = False,
    skip_topic: bool = False,
    skip_cooccurrence: bool = False,
    skip_report: bool = False,
) -> FullAnalysisOutputs:
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to 0")

    state_path = Path(state_file)
    state = load_workflow_state(state_path) if resume else empty_workflow_state()

    pipeline_signature = build_pipeline_signature(
        raw_data_dir=raw_data_dir,
        deduplicated_output_dir=deduplicated_output_dir,
        cleaned_output_dir=cleaned_output_dir,
    )
    expected_pipeline = expected_pipeline_outputs(
        raw_data_dir=raw_data_dir,
        deduplicated_output_dir=deduplicated_output_dir,
        cleaned_output_dir=cleaned_output_dir,
    )
    if resume and stage_is_complete(state, "pipeline", pipeline_signature, pipeline_output_paths(expected_pipeline)):
        pipeline_outputs = expected_pipeline
        print(f"Step 1/7: Resume hit for cleaned data, using {pipeline_outputs.cleaned_file}")
    else:
        print("Step 1/7: Deduplicate and clean data")
        pipeline_outputs = run_pipeline(
            raw_data_dir=raw_data_dir,
            deduplicated_output_dir=deduplicated_output_dir,
            cleaned_output_dir=cleaned_output_dir,
        )
        mark_stage_complete(
            state,
            "pipeline",
            pipeline_signature,
            {
                "deduplicated_file": pipeline_outputs.deduplicated_file,
                "cleaned_file": pipeline_outputs.cleaned_file,
                "flagged_file": pipeline_outputs.flagged_file,
            },
        )
        save_workflow_state(state_path, state)

    content_file = expected_content_file(content_output_dir)
    if skip_content:
        print(f"Step 2/7: Skipping content analysis, using {content_file}")
    else:
        content_signature = build_content_signature(
            cleaned_file=pipeline_outputs.cleaned_file,
            config_file=config_file,
            output_dir=content_output_dir,
            limit=limit,
        )
        if resume and not overwrite_llm and stage_is_complete(state, "content", content_signature, [content_file]):
            print(f"Step 2/7: Resume hit for content analysis, using {content_file}")
        else:
            print("Step 2/7: Run LLM content analysis")
            content_report = analyze_content(
                input_file=pipeline_outputs.cleaned_file,
                output_dir=content_output_dir,
                config_file=config_file,
                limit=limit,
                overwrite=overwrite_llm,
            )
            content_file = content_report.output_file
            if stage_report_succeeded(content_report):
                mark_stage_complete(
                    state,
                    "content",
                    content_signature,
                    {"content_file": content_file},
                )
                save_workflow_state(state_path, state)
            else:
                clear_stage(state, "content")
                save_workflow_state(state_path, state)

    sentiment_file = expected_sentiment_file(sentiment_output_dir)
    if skip_sentiment:
        print(f"Step 3/7: Skipping sentiment analysis, using {sentiment_file}")
    else:
        sentiment_signature = build_sentiment_signature(
            cleaned_file=pipeline_outputs.cleaned_file,
            config_file=config_file,
            output_dir=sentiment_output_dir,
            limit=limit,
        )
        if resume and not overwrite_llm and stage_is_complete(state, "sentiment", sentiment_signature, [sentiment_file]):
            print(f"Step 3/7: Resume hit for sentiment analysis, using {sentiment_file}")
        else:
            print("Step 3/7: Run LLM sentiment analysis")
            sentiment_report = analyze_sentiment(
                input_file=pipeline_outputs.cleaned_file,
                output_dir=sentiment_output_dir,
                config_file=config_file,
                limit=limit,
                overwrite=overwrite_llm,
            )
            sentiment_file = sentiment_report.output_file
            if stage_report_succeeded(sentiment_report):
                mark_stage_complete(
                    state,
                    "sentiment",
                    sentiment_signature,
                    {"sentiment_file": sentiment_file},
                )
                save_workflow_state(state_path, state)
            else:
                clear_stage(state, "sentiment")
                save_workflow_state(state_path, state)

    topic_file = Path(topic_output_dir) / "comment_topics.csv"
    topic_info_file = Path(topic_output_dir) / "topic_info.csv"
    if skip_topic:
        print(f"Step 4/7: Skipping topic clustering, using {topic_file}")
    else:
        topic_signature = build_topic_signature(
            cleaned_file=pipeline_outputs.cleaned_file,
            output_dir=topic_output_dir,
            limit=limit,
            min_text_length=min_text_length,
            embedding_model=embedding_model,
            min_topic_size=min_topic_size,
            save_topic_model=save_topic_model,
        )
        if resume and stage_is_complete(state, "topic", topic_signature, [topic_file, topic_info_file]):
            print(f"Step 4/7: Resume hit for topic clustering, using {topic_file}")
        else:
            print("Step 4/7: Run BERTopic clustering")
            topic_report = cluster_topics(
                input_file=pipeline_outputs.cleaned_file,
                output_dir=topic_output_dir,
                limit=limit,
                min_text_length=min_text_length,
                embedding_model=embedding_model,
                min_topic_size=min_topic_size,
                save_model=save_topic_model,
            )
            topic_file = topic_report.outputs.comment_topics_file
            topic_info_file = topic_report.outputs.topic_info_file
            mark_stage_complete(
                state,
                "topic",
                topic_signature,
                {
                    "topic_file": topic_file,
                    "topic_info_file": topic_info_file,
                    "representative_docs_file": topic_report.outputs.representative_docs_file,
                    "metadata_file": topic_report.outputs.metadata_file,
                    "model_dir": topic_report.outputs.model_dir,
                },
            )
            save_workflow_state(state_path, state)

    cooccurrence_summary_file = Path(cooccurrence_output_dir) / "summary.json"
    if skip_cooccurrence:
        print(f"Step 5/7: Skipping co-occurrence analysis, using {cooccurrence_summary_file}")
    else:
        cooccurrence_signature = build_cooccurrence_signature(
            cleaned_file=pipeline_outputs.cleaned_file,
            content_file=content_file,
            sentiment_file=sentiment_file,
            topic_file=topic_file,
            output_dir=cooccurrence_output_dir,
            limit=limit,
            min_confidence=min_confidence,
            min_sentiment_confidence=min_sentiment_confidence,
            min_edge_weight=min_edge_weight,
            include_fallback=include_fallback,
            include_sentiment_fallback=include_sentiment_fallback,
            include_neutral_polarity=include_neutral_polarity,
            include_topic_noise=include_topic_noise,
        )
        if resume and stage_is_complete(state, "cooccurrence", cooccurrence_signature, [cooccurrence_summary_file]):
            print(f"Step 5/7: Resume hit for co-occurrence analysis, using {cooccurrence_summary_file}")
        else:
            print("Step 5/7: Run NetworkX co-occurrence analysis")
            cooccurrence_report = analyze_cooccurrence(
                input_file=content_file,
                output_dir=cooccurrence_output_dir,
                cleaned_file=pipeline_outputs.cleaned_file,
                sentiment_file=sentiment_file,
                topic_file=topic_file,
                limit=limit,
                min_confidence=min_confidence,
                min_sentiment_confidence=min_sentiment_confidence,
                min_edge_weight=min_edge_weight,
                include_fallback=include_fallback,
                include_sentiment_fallback=include_sentiment_fallback,
                include_neutral_polarity=include_neutral_polarity,
                include_topic_noise=include_topic_noise,
            )
            cooccurrence_summary_file = cooccurrence_report.outputs.summary_file
            mark_stage_complete(
                state,
                "cooccurrence",
                cooccurrence_signature,
                {"summary_file": cooccurrence_summary_file},
            )
            save_workflow_state(state_path, state)

    report_html = Path(report_output_dir) / "report.html"
    interactive_report_html = Path(report_output_dir) / "interactive_report.html"
    report_metadata_file = Path(report_output_dir) / "report_metadata.json"
    if skip_report:
        print(f"Step 6/7: Skipping report generation, using {report_metadata_file}")
    else:
        report_signature = build_report_signature(
            cleaned_file=pipeline_outputs.cleaned_file,
            content_file=content_file,
            sentiment_file=sentiment_file,
            topic_file=topic_file,
            topic_info_file=topic_info_file,
            cooccurrence_summary_file=cooccurrence_summary_file,
            output_dir=report_output_dir,
            limit=limit,
            top_n=top_n,
            min_confidence=min_confidence,
            min_sentiment_confidence=min_sentiment_confidence,
            stopwords_file=stopwords_file,
        )
        if resume and stage_is_complete(
            state,
            "report",
            report_signature,
            [report_html, interactive_report_html, report_metadata_file],
        ):
            print(f"Step 6/7: Resume hit for report generation, using {report_metadata_file}")
        else:
            print("Step 6/7: Build summary tables and visual reports")
            report = build_analysis_report(
                cleaned_file=pipeline_outputs.cleaned_file,
                content_file=content_file,
                sentiment_file=sentiment_file,
                topic_file=topic_file,
                topic_info_file=topic_info_file,
                cooccurrence_dir=cooccurrence_output_dir,
                output_dir=report_output_dir,
                limit=limit,
                top_n=top_n,
                min_confidence=min_confidence,
                min_sentiment_confidence=min_sentiment_confidence,
                stopwords_file=stopwords_file,
            )
            report_html = report.static_html
            interactive_report_html = report.interactive_html
            report_metadata_file = report.metadata_file
            mark_stage_complete(
                state,
                "report",
                report_signature,
                {
                    "report_html": report_html,
                    "interactive_report_html": interactive_report_html,
                    "report_metadata_file": report_metadata_file,
                },
            )
            save_workflow_state(state_path, state)

    print("Step 7/7: Full analysis complete")
    print(f"Cleaned file: {pipeline_outputs.cleaned_file}")
    print(f"Content analysis: {content_file}")
    print(f"Sentiment analysis: {sentiment_file}")
    print(f"Topic clustering: {topic_file}")
    print(f"Co-occurrence summary: {cooccurrence_summary_file}")
    print(f"Report HTML: {report_html}")
    print(f"Interactive report HTML: {interactive_report_html}")

    return FullAnalysisOutputs(
        deduplicated_file=pipeline_outputs.deduplicated_file,
        cleaned_file=pipeline_outputs.cleaned_file,
        flagged_file=pipeline_outputs.flagged_file,
        content_file=Path(content_file),
        sentiment_file=Path(sentiment_file),
        topic_file=Path(topic_file),
        topic_info_file=Path(topic_info_file),
        cooccurrence_summary_file=Path(cooccurrence_summary_file),
        report_html=Path(report_html),
        interactive_report_html=Path(interactive_report_html),
        report_metadata_file=Path(report_metadata_file),
    )


def expected_content_file(output_dir: str | Path) -> Path:
    return Path(output_dir) / CONTENT_OUTPUT_FILE_NAME


def expected_sentiment_file(output_dir: str | Path) -> Path:
    return Path(output_dir) / SENTIMENT_OUTPUT_FILE_NAME


def expected_pipeline_outputs(
    *,
    raw_data_dir: str | Path,
    deduplicated_output_dir: str | Path | None,
    cleaned_output_dir: str | Path | None,
) -> PipelineOutputs:
    raw_path = Path(raw_data_dir)
    deduplicated_dir = (
        Path(deduplicated_output_dir)
        if deduplicated_output_dir is not None
        else default_stage_dir(raw_path, DEDUPLICATED_DIR_NAME)
    )
    cleaned_dir = (
        Path(cleaned_output_dir) if cleaned_output_dir is not None else default_stage_dir(raw_path, CLEANED_DIR_NAME)
    )
    return PipelineOutputs(
        deduplicated_file=deduplicated_dir / DEDUPLICATED_OUTPUT_FILE_NAME,
        cleaned_file=cleaned_dir / CLEANED_OUTPUT_FILE_NAME,
        flagged_file=cleaned_dir / DEFAULT_FLAGGED_FILE_NAME,
    )


def default_stage_dir(raw_data_dir: Path, stage_dir_name: str) -> Path:
    return raw_data_dir.parent / stage_dir_name if raw_data_dir.name == "raw" else raw_data_dir / stage_dir_name


def pipeline_output_paths(outputs: PipelineOutputs) -> list[Path]:
    return [outputs.deduplicated_file, outputs.cleaned_file, outputs.flagged_file]


def load_workflow_state(state_file: Path) -> dict[str, Any]:
    if not state_file.exists():
        return empty_workflow_state()
    try:
        with state_file.open("r", encoding="utf-8") as state_stream:
            state = json.load(state_stream)
    except (OSError, json.JSONDecodeError):
        return empty_workflow_state()
    if not isinstance(state, dict) or state.get("version") != STATE_VERSION or not isinstance(state.get("stages"), dict):
        return empty_workflow_state()
    return state


def empty_workflow_state() -> dict[str, Any]:
    return {"version": STATE_VERSION, "stages": {}}


def save_workflow_state(state_file: Path, state: dict[str, Any]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_is_complete(
    state: dict[str, Any],
    stage_name: str,
    signature: dict[str, Any],
    required_outputs: list[str | Path],
) -> bool:
    stage_state = state.get("stages", {}).get(stage_name)
    if not isinstance(stage_state, dict):
        return False
    if stage_state.get("status") != "ok" or stage_state.get("signature") != signature:
        return False
    return all(Path(output).exists() for output in required_outputs)


def mark_stage_complete(
    state: dict[str, Any],
    stage_name: str,
    signature: dict[str, Any],
    outputs: dict[str, str | Path | None],
) -> None:
    stages = state.setdefault("stages", {})
    stages[stage_name] = {
        "status": "ok",
        "signature": signature,
        "outputs": stringify_paths(outputs),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def clear_stage(state: dict[str, Any], stage_name: str) -> None:
    stages = state.setdefault("stages", {})
    stages.pop(stage_name, None)


def stringify_paths(values: dict[str, str | Path | None]) -> dict[str, str | None]:
    return {key: str(value) if value is not None else None for key, value in values.items()}


def stage_report_succeeded(report: object) -> bool:
    return int(getattr(report, "error_records", 0)) == 0


def build_pipeline_signature(
    *,
    raw_data_dir: str | Path,
    deduplicated_output_dir: str | Path | None,
    cleaned_output_dir: str | Path | None,
) -> dict[str, Any]:
    return {
        "raw_data_dir": path_signature(Path(raw_data_dir)),
        "deduplicated_output_dir": str(deduplicated_output_dir) if deduplicated_output_dir is not None else None,
        "cleaned_output_dir": str(cleaned_output_dir) if cleaned_output_dir is not None else None,
    }


def build_content_signature(
    *,
    cleaned_file: str | Path,
    config_file: str | Path,
    output_dir: str | Path,
    limit: int | None,
) -> dict[str, Any]:
    return {
        "cleaned_file": path_signature(Path(cleaned_file)),
        "config_file": path_signature(Path(config_file)),
        "output_dir": str(output_dir),
        "limit": limit,
    }


def build_sentiment_signature(
    *,
    cleaned_file: str | Path,
    config_file: str | Path,
    output_dir: str | Path,
    limit: int | None,
) -> dict[str, Any]:
    return {
        "cleaned_file": path_signature(Path(cleaned_file)),
        "config_file": path_signature(Path(config_file)),
        "output_dir": str(output_dir),
        "limit": limit,
    }


def build_topic_signature(
    *,
    cleaned_file: str | Path,
    output_dir: str | Path,
    limit: int | None,
    min_text_length: int,
    embedding_model: str,
    min_topic_size: int,
    save_topic_model: bool,
) -> dict[str, Any]:
    return {
        "cleaned_file": path_signature(Path(cleaned_file)),
        "output_dir": str(output_dir),
        "limit": limit,
        "min_text_length": min_text_length,
        "embedding_model": embedding_model,
        "min_topic_size": min_topic_size,
        "save_topic_model": save_topic_model,
    }


def build_cooccurrence_signature(
    *,
    cleaned_file: str | Path,
    content_file: str | Path,
    sentiment_file: str | Path,
    topic_file: str | Path,
    output_dir: str | Path,
    limit: int | None,
    min_confidence: float,
    min_sentiment_confidence: float | None,
    min_edge_weight: int,
    include_fallback: bool,
    include_sentiment_fallback: bool,
    include_neutral_polarity: bool,
    include_topic_noise: bool,
) -> dict[str, Any]:
    return {
        "cleaned_file": path_signature(Path(cleaned_file)),
        "content_file": path_signature(Path(content_file)),
        "sentiment_file": path_signature(Path(sentiment_file)),
        "topic_file": path_signature(Path(topic_file)),
        "output_dir": str(output_dir),
        "limit": limit,
        "min_confidence": min_confidence,
        "min_sentiment_confidence": min_sentiment_confidence,
        "min_edge_weight": min_edge_weight,
        "include_fallback": include_fallback,
        "include_sentiment_fallback": include_sentiment_fallback,
        "include_neutral_polarity": include_neutral_polarity,
        "include_topic_noise": include_topic_noise,
    }


def build_report_signature(
    *,
    cleaned_file: str | Path,
    content_file: str | Path,
    sentiment_file: str | Path,
    topic_file: str | Path,
    topic_info_file: str | Path,
    cooccurrence_summary_file: str | Path,
    output_dir: str | Path,
    limit: int | None,
    top_n: int,
    min_confidence: float,
    min_sentiment_confidence: float | None,
    stopwords_file: str | Path | None,
) -> dict[str, Any]:
    return {
        "cleaned_file": path_signature(Path(cleaned_file)),
        "content_file": path_signature(Path(content_file)),
        "sentiment_file": path_signature(Path(sentiment_file)),
        "topic_file": path_signature(Path(topic_file)),
        "topic_info_file": path_signature(Path(topic_info_file)),
        "cooccurrence_summary_file": path_signature(Path(cooccurrence_summary_file)),
        "output_dir": str(output_dir),
        "limit": limit,
        "top_n": top_n,
        "min_confidence": min_confidence,
        "min_sentiment_confidence": min_sentiment_confidence,
        "stopwords_file": path_signature(Path(stopwords_file)) if stopwords_file is not None else None,
    }


def path_signature(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_file():
        return {
            "path": str(path),
            "exists": True,
            "type": "file",
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if path.is_dir():
        files = []
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            files.append(
                {
                    "path": child.relative_to(path).as_posix(),
                    "size": child.stat().st_size,
                    "sha256": sha256_file(child),
                }
            )
        return {"path": str(path), "exists": True, "type": "directory", "files": files}
    return {"path": str(path), "exists": True, "type": "other"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_stream:
        for chunk in iter(lambda: file_stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full comment analysis workflow.")
    parser.add_argument("--raw-data-dir", default=str(DEFAULT_RAW_DATA_DIR))
    parser.add_argument("--deduplicated-output-dir", default=None)
    parser.add_argument("--cleaned-output-dir", default=None)
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG_FILE))
    parser.add_argument("--content-output-dir", default=str(DEFAULT_CONTENT_OUTPUT_DIR))
    parser.add_argument("--sentiment-output-dir", default=str(DEFAULT_SENTIMENT_OUTPUT_DIR))
    parser.add_argument("--topic-output-dir", default=str(DEFAULT_TOPIC_OUTPUT_DIR))
    parser.add_argument("--cooccurrence-output-dir", default=str(DEFAULT_COOCCURRENCE_OUTPUT_DIR))
    parser.add_argument("--report-output-dir", default=str(DEFAULT_REPORT_OUTPUT_DIR))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))
    parser.add_argument("--no-resume", action="store_true", help="Ignore workflow state and rerun every enabled stage.")
    parser.add_argument("--limit", type=int, default=None, help="Limit analysis stages to the first N cleaned records.")
    parser.add_argument("--overwrite-llm", action="store_true", help="Clear content/sentiment JSONL before LLM analysis.")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-sentiment-confidence", type=float, default=None)
    parser.add_argument("--min-edge-weight", type=int, default=1)
    parser.add_argument("--include-fallback", action="store_true")
    parser.add_argument("--include-sentiment-fallback", action="store_true")
    parser.add_argument("--exclude-neutral-polarity", action="store_true")
    parser.add_argument("--include-topic-noise", action="store_true")
    parser.add_argument("--min-text-length", type=int, default=DEFAULT_MIN_TEXT_LENGTH)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--min-topic-size", type=int, default=DEFAULT_MIN_TOPIC_SIZE)
    parser.add_argument("--save-topic-model", action="store_true")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--stopwords-file", default=None)
    parser.add_argument("--skip-content", action="store_true")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--skip-topic", action="store_true")
    parser.add_argument("--skip-cooccurrence", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    run_full_analysis(
        raw_data_dir=args.raw_data_dir,
        deduplicated_output_dir=args.deduplicated_output_dir,
        cleaned_output_dir=args.cleaned_output_dir,
        config_file=args.config_file,
        content_output_dir=args.content_output_dir,
        sentiment_output_dir=args.sentiment_output_dir,
        topic_output_dir=args.topic_output_dir,
        cooccurrence_output_dir=args.cooccurrence_output_dir,
        report_output_dir=args.report_output_dir,
        state_file=args.state_file,
        resume=not args.no_resume,
        limit=args.limit,
        overwrite_llm=args.overwrite_llm,
        min_confidence=args.min_confidence,
        min_sentiment_confidence=args.min_sentiment_confidence,
        min_edge_weight=args.min_edge_weight,
        include_fallback=args.include_fallback,
        include_sentiment_fallback=args.include_sentiment_fallback,
        include_neutral_polarity=not args.exclude_neutral_polarity,
        include_topic_noise=args.include_topic_noise,
        min_text_length=args.min_text_length,
        embedding_model=args.embedding_model,
        min_topic_size=args.min_topic_size,
        save_topic_model=args.save_topic_model,
        top_n=args.top_n,
        stopwords_file=args.stopwords_file,
        skip_content=args.skip_content,
        skip_sentiment=args.skip_sentiment,
        skip_topic=args.skip_topic,
        skip_cooccurrence=args.skip_cooccurrence,
        skip_report=args.skip_report,
    )


if __name__ == "__main__":
    main()
