import csv

from dataloader import CommentRecord
from scripts.deduplicate_data import deduplicate_data, deduplicate_records


def test_deduplicate_records_keeps_first_occurrence():
    first = CommentRecord(
        username="Alice",
        gender="",
        content="same",
        comment_time="2026-05-10 10:00:00",
        likes=1,
        ip_location="Shanghai",
        signature="",
        feature="aigc",
    )
    second = CommentRecord(
        username="Bob",
        gender="",
        content="different",
        comment_time="2026-05-10 10:01:00",
        likes=2,
        ip_location="Beijing",
        signature="",
        feature="aigc",
    )

    assert deduplicate_records([first, second, first]) == [first, second]


def test_deduplicate_data_reads_raw_and_writes_sibling_output(tmp_path):
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)

    source_file = raw_dir / "comments.csv"
    source_file.write_text(
        "用户名,性别,评论内容,评论时间,点赞数,IP属地,个性签名\n"
        "Alice,保密,hello,2026-05-10 10:00:00,3,上海,signature\n"
        "Alice,保密,hello,2026-05-10 10:00:00,3,上海,signature\n",
        encoding="utf-8",
    )

    output_file = deduplicate_data(raw_dir)

    with output_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert output_file == data_dir / "deduplicated" / "comments_deduplicated.csv"
    assert len(rows) == 1
    assert rows[0]["username"] == "Alice"
