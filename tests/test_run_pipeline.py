import csv

from scripts.run_pipeline import run_pipeline


def test_run_pipeline_deduplicates_then_cleans_raw_data(tmp_path):
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    raw_file = raw_dir / "comments.csv"

    with raw_file.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["feature", "nickname", "content", "create_time", "like_count", "ip_location"],
        )
        writer.writeheader()
        row = {
            "feature": "aigc",
            "nickname": "Alice",
            "content": " hello ",
            "create_time": "1720946095000",
            "like_count": "3",
            "ip_location": "",
        }
        writer.writerow(row)
        writer.writerow(row)

    outputs = run_pipeline(raw_dir)

    assert outputs.deduplicated_file == data_dir / "deduplicated" / "comments_deduplicated.csv"
    assert outputs.cleaned_file == data_dir / "cleaned" / "comments_cleaned.csv"
    assert outputs.flagged_file == data_dir / "cleaned" / "rejected_or_flagged_comments.csv"

    with outputs.cleaned_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
        cleaned_rows = list(csv.DictReader(csv_file))
    with outputs.flagged_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
        flagged_rows = list(csv.DictReader(csv_file))

    assert len(cleaned_rows) == 1
    assert cleaned_rows[0]["username"] == "Alice"
    assert cleaned_rows[0]["content"] == "hello"
    assert cleaned_rows[0]["comment_time"] == "2024-07-14 16:34:55"
    assert flagged_rows == []
