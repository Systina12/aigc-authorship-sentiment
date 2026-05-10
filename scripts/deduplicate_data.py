from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloader import CommentRecord, DataLoader


DEFAULT_OUTPUT_DIR_NAME = "deduplicated"
DEFAULT_OUTPUT_FILE_NAME = "comments_deduplicated.csv"


def deduplicate_records(records: list[CommentRecord]) -> list[CommentRecord]:
    seen: set[tuple[object, ...]] = set()
    deduplicated: list[CommentRecord] = []

    for record in records:
        key = tuple(getattr(record, field.name) for field in fields(CommentRecord))
        if key in seen:
            continue

        seen.add(key)
        deduplicated.append(record)

    return deduplicated


def write_records(records: list[CommentRecord], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(CommentRecord)]

    with output_file.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def deduplicate_data(data_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    data_path = Path(data_dir)
    output_path = (
        Path(output_dir)
        if output_dir is not None
        else (data_path.parent / DEFAULT_OUTPUT_DIR_NAME if data_path.name == "raw" else data_path / DEFAULT_OUTPUT_DIR_NAME)
    )
    output_file = output_path / DEFAULT_OUTPUT_FILE_NAME

    records = DataLoader().load(data_path)
    deduplicated = deduplicate_records(records)
    write_records(deduplicated, output_file)

    print(f"Loaded records: {len(records)}")
    print(f"Deduplicated records: {len(deduplicated)}")
    print(f"Removed duplicates: {len(records) - len(deduplicated)}")
    print(f"Output file: {output_file}")

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate loaded comment records.")
    parser.add_argument("--data-dir", default="data/raw", help="Folder containing source data files.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder for deduplicated output. Defaults to data/deduplicated when reading data/raw.",
    )
    args = parser.parse_args()

    deduplicate_data(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
