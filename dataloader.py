from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Protocol


@dataclass(frozen=True)
class CommentRecord:
    username: str
    gender: str
    content: str
    comment_time: str
    likes: int
    ip_location: str
    signature: str
    feature: str = ""


class UnsupportedFileError(ValueError):
    pass


class DataParser(Protocol):
    database_name: str

    def matches(self, fieldnames: set[str]) -> bool:
        ...

    def parse(self, file_path: Path) -> list[CommentRecord]:
        ...


class BilibiliCommentScrapeParser:
    database_name = "BilibiliCommentScrape"

    _field_mapping = {
        "用户名": "username",
        "性别": "gender",
        "评论内容": "content",
        "评论时间": "comment_time",
        "点赞数": "likes",
        "IP属地": "ip_location",
        "个性签名": "signature",
    }

    def matches(self, fieldnames: set[str]) -> bool:
        return set(self._field_mapping).issubset(fieldnames)

    def parse(self, file_path: Path) -> list[CommentRecord]:
        records: list[CommentRecord] = []

        with _open_csv(file_path) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                records.append(
                    CommentRecord(
                        username=row["用户名"],
                        gender=row["性别"],
                        content=row["评论内容"],
                        comment_time=row["评论时间"],
                        likes=_parse_int(row["点赞数"], field_name="点赞数", file_path=file_path),
                        ip_location=row["IP属地"],
                        signature=row["个性签名"],
                    )
                )

        return records


class XiaohongshuAIGCParser:
    database_name = "Xiaohongshu-AIGC"

    _field_mapping = {
        "feature": "feature",
        "nickname": "username",
        "content": "content",
        "create_time": "comment_time",
        "like_count": "likes",
        "ip_location": "ip_location",
    }

    def matches(self, fieldnames: set[str]) -> bool:
        return set(self._field_mapping).issubset(fieldnames)

    def parse(self, file_path: Path) -> list[CommentRecord]:
        records: list[CommentRecord] = []

        with _open_csv(file_path) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                records.append(
                    CommentRecord(
                        username=row["nickname"],
                        gender="",
                        content=row["content"],
                        comment_time=row["create_time"],
                        likes=_parse_int(row["like_count"], field_name="like_count", file_path=file_path),
                        ip_location=row["ip_location"],
                        signature="",
                        feature=row["feature"],
                    )
                )

        return records


class CommentRecordCSVParser:
    database_name = "CommentRecordCSV"

    _required_fields = {"username", "gender", "content", "comment_time", "likes", "ip_location", "signature"}

    def matches(self, fieldnames: set[str]) -> bool:
        return self._required_fields.issubset(fieldnames)

    def parse(self, file_path: Path) -> list[CommentRecord]:
        records: list[CommentRecord] = []

        with _open_csv(file_path) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                records.append(
                    CommentRecord(
                        username=row["username"],
                        gender=row["gender"],
                        content=row["content"],
                        comment_time=row["comment_time"],
                        likes=_parse_int(row["likes"], field_name="likes", file_path=file_path),
                        ip_location=row["ip_location"],
                        signature=row["signature"],
                        feature=row.get("feature", ""),
                    )
                )

        return records


class DataLoader:
    def __init__(self, parsers: Iterable[DataParser] | None = None) -> None:
        self._parsers = list(
            parsers or [BilibiliCommentScrapeParser(), XiaohongshuAIGCParser(), CommentRecordCSVParser()]
        )

    def load(self, path: str | Path) -> list[CommentRecord]:
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        if data_path.is_file():
            parser = self._find_parser(data_path)
            return parser.parse(data_path)
        if not data_path.is_dir():
            raise NotADirectoryError(f"Data path is not a folder or file: {data_path}")

        records: list[CommentRecord] = []
        for file_path in sorted(path for path in data_path.rglob("*") if path.is_file()):
            parser = self._find_parser(file_path)
            records.extend(parser.parse(file_path))

        return records

    def _find_parser(self, file_path: Path) -> DataParser:
        if file_path.suffix.lower() != ".csv":
            raise UnsupportedFileError(f"Unsupported file type: {file_path}")

        fieldnames = _read_csv_fieldnames(file_path)
        for parser in self._parsers:
            if parser.matches(fieldnames):
                return parser

        raise UnsupportedFileError(f"Unrecognized data schema: {file_path}")


def load_data(folder: str | Path) -> list[CommentRecord]:
    return DataLoader().load(folder)


def _read_csv_fieldnames(file_path: Path) -> set[str]:
    with _open_csv(file_path) as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise UnsupportedFileError(f"CSV has no header row: {file_path}")
        return set(reader.fieldnames)


def _open_csv(file_path: Path):
    encoding_options = (
        ("utf-8-sig", "strict"),
        ("gb18030", "replace"),
    )

    for encoding, errors in encoding_options:
        csv_file = file_path.open("r", encoding=encoding, errors=errors, newline="")
        try:
            csv_file.read(4096)
            csv_file.seek(0)
            return csv_file
        except UnicodeDecodeError:
            csv_file.close()

    raise UnicodeDecodeError("csv", b"", 0, 1, f"Unable to decode CSV: {file_path}")


def _parse_int(value: str, *, field_name: str, file_path: Path) -> int:
    normalized = value.strip().replace(",", "")
    unit_multipliers = {
        "万": Decimal("10000"),
        "千": Decimal("1000"),
    }

    for unit, multiplier in unit_multipliers.items():
        if normalized.endswith(unit):
            number = normalized[: -len(unit)]
            try:
                return int(Decimal(number) * multiplier)
            except InvalidOperation as exc:
                raise ValueError(f"Invalid integer in {field_name} for {file_path}: {value!r}") from exc

    try:
        return int(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid integer in {field_name} for {file_path}: {value!r}") from exc
