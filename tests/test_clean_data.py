import csv

from dataloader import CommentRecord
from scripts.clean_data import DEFAULT_FLAGGED_FILE_NAME, clean_data, clean_record, clean_records, clean_records_with_flags


def test_clean_record_strips_text_and_normalizes_null_markers():
    record = CommentRecord(
        username=" Alice ",
        gender=" nan ",
        content=" hello ",
        comment_time=" 2026-05-10 10:00:00 ",
        likes=1,
        ip_location=" null ",
        signature=" N/A ",
        feature=" ai-advertisement ",
    )

    cleaned, issues = clean_record(record)

    assert cleaned == CommentRecord(
        username="Alice",
        gender="未知",
        content="hello",
        comment_time="2026-05-10 10:00:00",
        likes=1,
        ip_location="未知",
        signature="",
        feature="ai-advertisement",
    )
    assert issues.empty_content == 0
    assert issues.invalid_time == 0


def test_clean_records_removes_empty_content():
    records = [
        CommentRecord("Alice", "", " hello ", "2026-05-10 10:00:00", 1, "上海", "", ""),
        CommentRecord("Bob", "", " null ", "2026-05-10 10:00:00", 1, "上海", "", ""),
    ]

    cleaned, report = clean_records(records)

    assert len(cleaned) == 1
    assert cleaned[0].username == "Alice"
    assert report.loaded_records == 2
    assert report.cleaned_records == 1
    assert report.empty_content_records == 1


def test_clean_records_with_flags_keeps_flagged_rows_and_removes_empty_content():
    records = [
        CommentRecord("Empty", "", " null ", "2026-05-10 10:00:00", 1, "", "", ""),
        CommentRecord("BadTime", "", "hello", "bad-time", 1, "", "", ""),
        CommentRecord("Negative", "", "hello", "2026-05-10 10:00:00", -1, "", "", ""),
    ]

    cleaned, flagged, report = clean_records_with_flags(records)

    assert [record.username for record in cleaned] == ["BadTime", "Negative"]
    assert [(record.record.username, record.issue_types) for record in flagged] == [
        ("Empty", ["empty_content"]),
        ("BadTime", ["invalid_time"]),
        ("Negative", ["negative_likes"]),
    ]
    assert report.cleaned_records == 2
    assert report.flagged_records == 3
    assert report.empty_content_records == 1
    assert report.invalid_time_records == 1
    assert report.negative_likes_records == 1


def test_clean_record_converts_millisecond_timestamp_to_shanghai_time():
    record = CommentRecord("Alice", "", "hello", "1720946095000", 1, "", "", "")

    cleaned, issues = clean_record(record)

    assert cleaned.comment_time == "2024-07-14 16:34:55"
    assert issues.invalid_time == 0


def test_clean_record_keeps_invalid_time_and_counts_it():
    record = CommentRecord("Alice", "", "hello", "not-a-time", 1, "", "", "")

    cleaned, issues = clean_record(record)

    assert cleaned.comment_time == "not-a-time"
    assert issues.invalid_time == 1


def test_clean_record_normalizes_categories_and_counts_negative_likes():
    record = CommentRecord("Alice", " 女 ", "hello", "2026-05-10 10:00:00", -1, " ", "", "aigc")

    cleaned, issues = clean_record(record)

    assert cleaned.gender == "女"
    assert cleaned.ip_location == "未知"
    assert cleaned.feature == "aigc"
    assert cleaned.likes == -1
    assert issues.negative_likes == 1


def test_clean_data_reads_input_and_writes_cleaned_output(tmp_path):
    input_file = tmp_path / "comments_deduplicated.csv"
    with input_file.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["username", "gender", "content", "comment_time", "likes", "ip_location", "signature", "feature"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "username": " Alice ",
                "gender": "保密",
                "content": " hello ",
                "comment_time": "1720946095",
                "likes": "2",
                "ip_location": "",
                "signature": " none ",
                "feature": " aigc ",
            }
        )

    output_file = clean_data(input_file, tmp_path / "cleaned")

    with output_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert output_file == tmp_path / "cleaned" / "comments_cleaned.csv"
    assert rows == [
        {
            "username": "Alice",
            "gender": "保密",
            "content": "hello",
            "comment_time": "2024-07-14 16:34:55",
            "likes": "2",
            "ip_location": "未知",
            "signature": "",
            "feature": "aigc",
        }
    ]


def test_clean_data_writes_rejected_or_flagged_output(tmp_path):
    input_file = tmp_path / "comments_deduplicated.csv"
    with input_file.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["username", "gender", "content", "comment_time", "likes", "ip_location", "signature", "feature"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "username": "Empty",
                "gender": "",
                "content": " null ",
                "comment_time": "2026-05-10 10:00:00",
                "likes": "1",
                "ip_location": "",
                "signature": "",
                "feature": "",
            }
        )
        writer.writerow(
            {
                "username": "Bad",
                "gender": "",
                "content": "hello",
                "comment_time": "bad-time",
                "likes": "-1",
                "ip_location": "",
                "signature": "",
                "feature": "",
            }
        )

    output_dir = tmp_path / "cleaned"
    output_file = clean_data(input_file, output_dir)
    flagged_file = output_dir / DEFAULT_FLAGGED_FILE_NAME

    with output_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
        cleaned_rows = list(csv.DictReader(csv_file))
    with flagged_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
        flagged_rows = list(csv.DictReader(csv_file))

    assert [row["username"] for row in cleaned_rows] == ["Bad"]
    assert [(row["username"], row["issue_types"]) for row in flagged_rows] == [
        ("Empty", "empty_content"),
        ("Bad", "invalid_time;negative_likes"),
    ]
