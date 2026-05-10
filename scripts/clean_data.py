from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloader import CommentRecord
from scripts.deduplicate_data import write_records


DEFAULT_INPUT_FILE = Path("data/deduplicated/comments_deduplicated.csv")
DEFAULT_OUTPUT_DIR = Path("data/cleaned")
DEFAULT_OUTPUT_FILE_NAME = "comments_cleaned.csv"
DEFAULT_FLAGGED_FILE_NAME = "rejected_or_flagged_comments.csv"
SHANGHAI_TZ = timezone(timedelta(hours=8))

NULL_MARKERS = {"", "nan", "none", "null", "n/a", "na"}
TEXT_FIELDS = ("username", "gender", "content", "comment_time", "ip_location", "signature", "feature")
STANDARD_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

MALE_VALUES = {"男", "male", "m"}
FEMALE_VALUES = {"女", "female", "f"}
PRIVATE_VALUES = {"保密", "秘密", "private", "hidden", "unknown gender"}
UNKNOWN_VALUES = {"未知", "unknown", "unk"}


@dataclass(frozen=True)
class CleanIssueStats:
    empty_content: int = 0
    invalid_time: int = 0
    negative_likes: int = 0

    def issue_types(self) -> list[str]:
        issue_types: list[str] = []
        if self.empty_content:
            issue_types.append("empty_content")
        if self.invalid_time:
            issue_types.append("invalid_time")
        if self.negative_likes:
            issue_types.append("negative_likes")

        return issue_types

    def __add__(self, other: CleanIssueStats) -> CleanIssueStats:
        return CleanIssueStats(
            empty_content=self.empty_content + other.empty_content,
            invalid_time=self.invalid_time + other.invalid_time,
            negative_likes=self.negative_likes + other.negative_likes,
        )


@dataclass(frozen=True)
class CleanReport:
    loaded_records: int
    cleaned_records: int
    flagged_records: int
    empty_content_records: int
    invalid_time_records: int
    negative_likes_records: int


@dataclass(frozen=True)
class FlaggedRecord:
    record: CommentRecord
    issue_types: list[str]


def clean_record(record: CommentRecord) -> tuple[CommentRecord, CleanIssueStats]:
    cleaned_values = {field: _clean_text(getattr(record, field)) for field in TEXT_FIELDS}
    comment_time, invalid_time = _normalize_comment_time(cleaned_values["comment_time"])

    cleaned = CommentRecord(
        username=cleaned_values["username"],
        gender=_normalize_gender(cleaned_values["gender"]),
        content=cleaned_values["content"],
        comment_time=comment_time,
        likes=record.likes,
        ip_location=cleaned_values["ip_location"] or "未知",
        signature=cleaned_values["signature"],
        feature=cleaned_values["feature"],
    )
    issues = CleanIssueStats(
        empty_content=1 if cleaned.content == "" else 0,
        invalid_time=1 if invalid_time else 0,
        negative_likes=1 if cleaned.likes < 0 else 0,
    )

    return cleaned, issues


def clean_records(records: list[CommentRecord]) -> tuple[list[CommentRecord], CleanReport]:
    cleaned_records, _, report = clean_records_with_flags(records)
    return cleaned_records, report


def clean_records_with_flags(records: list[CommentRecord]) -> tuple[list[CommentRecord], list[FlaggedRecord], CleanReport]:
    cleaned_records: list[CommentRecord] = []
    flagged_records: list[FlaggedRecord] = []
    issue_totals = CleanIssueStats()

    for record in records:
        cleaned, issues = clean_record(record)
        issue_totals += issues
        issue_types = issues.issue_types()
        if issue_types:
            flagged_records.append(FlaggedRecord(record=cleaned, issue_types=issue_types))

        if issues.empty_content:
            continue

        cleaned_records.append(cleaned)

    return cleaned_records, flagged_records, CleanReport(
        loaded_records=len(records),
        cleaned_records=len(cleaned_records),
        flagged_records=len(flagged_records),
        empty_content_records=issue_totals.empty_content,
        invalid_time_records=issue_totals.invalid_time,
        negative_likes_records=issue_totals.negative_likes,
    )


def clean_data(input_file: str | Path = DEFAULT_INPUT_FILE, output_dir: str | Path | None = None) -> Path:
    input_path = Path(input_file)
    output_path = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output_file = output_path / DEFAULT_OUTPUT_FILE_NAME
    flagged_file = output_path / DEFAULT_FLAGGED_FILE_NAME

    records = read_records(input_path)
    cleaned_records, flagged_records, report = clean_records_with_flags(records)
    write_records(cleaned_records, output_file)
    write_flagged_records(flagged_records, flagged_file)

    print(f"Loaded records: {report.loaded_records}")
    print(f"Cleaned records: {report.cleaned_records}")
    print(f"Rejected or flagged records: {report.flagged_records}")
    print(f"Removed empty-content records: {report.empty_content_records}")
    print(f"Invalid time records kept: {report.invalid_time_records}")
    print(f"Negative likes records kept: {report.negative_likes_records}")
    print(f"Output file: {output_file}")
    print(f"Rejected or flagged file: {flagged_file}")

    return output_file


def read_records(input_file: Path) -> list[CommentRecord]:
    with input_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        records: list[CommentRecord] = []
        for row in reader:
            records.append(
                CommentRecord(
                    username=row["username"],
                    gender=row["gender"],
                    content=row["content"],
                    comment_time=row["comment_time"],
                    likes=int(row["likes"]),
                    ip_location=row["ip_location"],
                    signature=row["signature"],
                    feature=row.get("feature", ""),
                )
            )

    return records


def write_flagged_records(flagged_records: list[FlaggedRecord], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    record_fieldnames = [field.name for field in fields(CommentRecord)]
    fieldnames = ["issue_types", *record_fieldnames]

    with output_file.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for flagged_record in flagged_records:
            writer.writerow(
                {
                    "issue_types": ";".join(flagged_record.issue_types),
                    **asdict(flagged_record.record),
                }
            )


def _clean_text(value: object) -> str:
    normalized = str(value).strip()
    if normalized.casefold() in NULL_MARKERS:
        return ""

    return normalized


def _normalize_comment_time(value: str) -> tuple[str, bool]:
    if value == "":
        return value, True

    parsed_standard = _parse_standard_time(value)
    if parsed_standard is not None:
        return parsed_standard.strftime(STANDARD_TIME_FORMAT), False

    if value.isdigit() and len(value) in {10, 13}:
        timestamp = int(value)
        if len(value) == 13:
            timestamp = timestamp / 1000
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone(SHANGHAI_TZ).strftime(
                STANDARD_TIME_FORMAT
            ), False
        except (OverflowError, OSError, ValueError):
            return value, True

    return value, True


def _parse_standard_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, STANDARD_TIME_FORMAT)
    except ValueError:
        return None


def _normalize_gender(value: str) -> str:
    if value == "":
        return "未知"

    normalized = value.casefold()
    if value in MALE_VALUES or normalized in MALE_VALUES:
        return "男"
    if value in FEMALE_VALUES or normalized in FEMALE_VALUES:
        return "女"
    if value in PRIVATE_VALUES or normalized in PRIVATE_VALUES:
        return "保密"
    if value in UNKNOWN_VALUES or normalized in UNKNOWN_VALUES:
        return "未知"

    return "未知"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean deduplicated comment records.")
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT_FILE),
        help="Deduplicated CSV file to clean. Defaults to data/deduplicated/comments_deduplicated.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder for cleaned output. Defaults to data/cleaned.",
    )
    args = parser.parse_args()

    clean_data(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
