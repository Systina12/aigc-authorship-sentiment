import csv

import pytest

from dataloader import CommentRecord, UnsupportedFileError, load_data


def test_load_bilibili_comment_scrape_csv(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "comments.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "序号",
                "用户名",
                "性别",
                "评论内容",
                "评论时间",
                "点赞数",
                "IP属地",
                "个性签名",
                "头像",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "序号": "1",
                "用户名": "测试用户",
                "性别": "女",
                "评论内容": "这是一条评论",
                "评论时间": "2026-05-08 21:40:42",
                "点赞数": "12",
                "IP属地": "湖南",
                "个性签名": "测试签名",
                "头像": "https://example.test/avatar.jpg",
            }
        )

    assert load_data(data_dir) == [
        CommentRecord(
            username="测试用户",
            gender="女",
            content="这是一条评论",
            comment_time="2026-05-08 21:40:42",
            likes=12,
            ip_location="湖南",
            signature="测试签名",
        )
    ]


def test_load_walks_nested_files(tmp_path):
    nested_dir = tmp_path / "data" / "nested"
    nested_dir.mkdir(parents=True)
    csv_path = nested_dir / "comments.csv"

    csv_path.write_text(
        "用户名,性别,评论内容,评论时间,点赞数,IP属地,个性签名\n"
        "Alice,保密,hello,2026-05-10 10:00:00,3,广东,signature\n",
        encoding="utf-8",
    )

    assert len(load_data(tmp_path / "data")) == 1


def test_load_xiaohongshu_aigc_csv(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "Comment.csv"

    csv_path.write_text(
        "feature,comment_id,create_time,ip_location,note_id,content,user_id,nickname,"
        "avatar,sub_comment_count,pictures,parent_comment_id,last_modify_ts,like_count\n"
        "ai-advertisement,66938daf000000001303e4dd,1720946095000,上海,note-1,"
        "做一张海报下载免费吗,user-1,敏敏的速写本,avatar.jpg,3,,0,1728904545276,1.5万\n",
        encoding="gb18030",
    )

    assert load_data(data_dir) == [
        CommentRecord(
            username="敏敏的速写本",
            gender="",
            content="做一张海报下载免费吗",
            comment_time="1720946095000",
            likes=15000,
            ip_location="上海",
            signature="",
            feature="ai-advertisement",
        )
    ]


def test_load_standard_comment_record_csv_file(tmp_path):
    csv_path = tmp_path / "comments_cleaned.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["username", "gender", "content", "comment_time", "likes", "ip_location", "signature", "feature"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "username": "Alice",
                "gender": "保密",
                "content": "AI 工具能提高效率",
                "comment_time": "2026-05-10 10:00:00",
                "likes": "12",
                "ip_location": "上海",
                "signature": "",
                "feature": "aigc",
            }
        )

    assert load_data(csv_path) == [
        CommentRecord(
            username="Alice",
            gender="保密",
            content="AI 工具能提高效率",
            comment_time="2026-05-10 10:00:00",
            likes=12,
            ip_location="上海",
            signature="",
            feature="aigc",
        )
    ]


def test_unrecognized_schema_raises(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "unknown.csv").write_text("name,message\nAlice,hello\n", encoding="utf-8")

    with pytest.raises(UnsupportedFileError):
        load_data(data_dir)
