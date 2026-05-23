from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

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
from scripts.cluster_topics import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MIN_TEXT_LENGTH,
    DEFAULT_MIN_TOPIC_SIZE,
    DEFAULT_OUTPUT_DIR as DEFAULT_TOPIC_OUTPUT_DIR,
    cluster_topics,
)
from scripts.run_pipeline import DEFAULT_RAW_DATA_DIR, run_pipeline


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

    print("Step 1/7: Deduplicate and clean data")
    pipeline_outputs = run_pipeline(
        raw_data_dir=raw_data_dir,
        deduplicated_output_dir=deduplicated_output_dir,
        cleaned_output_dir=cleaned_output_dir,
    )

    content_file = expected_content_file(content_output_dir)
    if skip_content:
        print(f"Step 2/7: Skipping content analysis, using {content_file}")
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

    sentiment_file = expected_sentiment_file(sentiment_output_dir)
    if skip_sentiment:
        print(f"Step 3/7: Skipping sentiment analysis, using {sentiment_file}")
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

    topic_file = Path(topic_output_dir) / "comment_topics.csv"
    topic_info_file = Path(topic_output_dir) / "topic_info.csv"
    if skip_topic:
        print(f"Step 4/7: Skipping topic clustering, using {topic_file}")
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

    cooccurrence_summary_file = Path(cooccurrence_output_dir) / "summary.json"
    if skip_cooccurrence:
        print(f"Step 5/7: Skipping co-occurrence analysis, using {cooccurrence_summary_file}")
    else:
        print("Step 5/7: Run NetworkX co-occurrence analysis")
        cooccurrence_report = analyze_cooccurrence(
            input_file=content_file,
            output_dir=cooccurrence_output_dir,
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

    report_html = Path(report_output_dir) / "report.html"
    interactive_report_html = Path(report_output_dir) / "interactive_report.html"
    report_metadata_file = Path(report_output_dir) / "report_metadata.json"
    if skip_report:
        print(f"Step 6/7: Skipping report generation, using {report_metadata_file}")
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
