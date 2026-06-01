from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_content import DEFAULT_OUTPUT_FILE as DEFAULT_CONTENT_FILE
from scripts.analyze_sentiment import DEFAULT_OUTPUT_FILE as DEFAULT_SENTIMENT_FILE


DEFAULT_TARGET_FILES = (DEFAULT_CONTENT_FILE, DEFAULT_SENTIMENT_FILE)


@dataclass(frozen=True)
class CleanErrorsReport:
    input_file: Path
    output_file: Path
    backup_file: Path | None
    total_rows: int
    kept_ok_rows: int
    removed_error_rows: int
    removed_invalid_rows: int
    dry_run: bool


def clean_llm_error_rows(
    input_file: str | Path,
    *,
    output_file: str | Path | None = None,
    backup: bool = True,
    dry_run: bool = False,
) -> CleanErrorsReport:
    input_path = Path(input_file)
    output_path = Path(output_file) if output_file is not None else input_path
    if not input_path.exists():
        return CleanErrorsReport(
            input_file=input_path,
            output_file=output_path,
            backup_file=None,
            total_rows=0,
            kept_ok_rows=0,
            removed_error_rows=0,
            removed_invalid_rows=0,
            dry_run=dry_run,
        )

    kept_rows: list[dict[str, Any]] = []
    total_rows = 0
    removed_error_rows = 0
    removed_invalid_rows = 0

    with input_path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if not line.strip():
                continue
            total_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                removed_invalid_rows += 1
                continue

            if row.get("status") == "ok":
                kept_rows.append(row)
            else:
                removed_error_rows += 1

    backup_file = None
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path == input_path and backup:
            backup_file = next_backup_path(input_path)
            backup_file.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
        write_jsonl(output_path, kept_rows)

    return CleanErrorsReport(
        input_file=input_path,
        output_file=output_path,
        backup_file=backup_file,
        total_rows=total_rows,
        kept_ok_rows=len(kept_rows),
        removed_error_rows=removed_error_rows,
        removed_invalid_rows=removed_invalid_rows,
        dry_run=dry_run,
    )


def next_backup_path(input_file: Path) -> Path:
    candidate = input_file.with_suffix(input_file.suffix + ".bak")
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = input_file.with_suffix(input_file.suffix + f".bak{index}")
        if not candidate.exists():
            return candidate
        index += 1


def write_jsonl(output_file: Path, rows: list[dict[str, Any]]) -> None:
    with output_file.open("w", encoding="utf-8", newline="\n") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_report(report: CleanErrorsReport) -> None:
    print(f"File: {report.input_file}")
    print(f"Total rows: {report.total_rows}")
    print(f"Kept ok rows: {report.kept_ok_rows}")
    print(f"Removed error rows: {report.removed_error_rows}")
    print(f"Removed invalid rows: {report.removed_invalid_rows}")
    print(f"Output file: {report.output_file}")
    print(f"Backup file: {report.backup_file if report.backup_file is not None else '(none)'}")
    if report.dry_run:
        print("Dry run: no files were changed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove non-ok rows from LLM JSONL result files.")
    parser.add_argument(
        "files",
        nargs="*",
        default=[str(path) for path in DEFAULT_TARGET_FILES],
        help="JSONL files to clean. Defaults to content and sentiment analysis outputs.",
    )
    parser.add_argument("--output-file", default=None, help="Write one input file to a separate output file.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a .bak copy before in-place cleaning.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be removed.")
    args = parser.parse_args()

    if args.output_file is not None and len(args.files) != 1:
        raise SystemExit("--output-file can only be used with exactly one input file")

    for file_name in args.files:
        report = clean_llm_error_rows(
            file_name,
            output_file=args.output_file,
            backup=not args.no_backup,
            dry_run=args.dry_run,
        )
        print_report(report)


if __name__ == "__main__":
    main()
