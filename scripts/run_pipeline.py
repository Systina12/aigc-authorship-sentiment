from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clean_data import DEFAULT_FLAGGED_FILE_NAME, clean_data
from scripts.deduplicate_data import deduplicate_data


DEFAULT_RAW_DATA_DIR = Path("data/raw")
DEDUPLICATED_DIR_NAME = "deduplicated"
CLEANED_DIR_NAME = "cleaned"


@dataclass(frozen=True)
class PipelineOutputs:
    deduplicated_file: Path
    cleaned_file: Path
    flagged_file: Path


def run_pipeline(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    deduplicated_output_dir: str | Path | None = None,
    cleaned_output_dir: str | Path | None = None,
) -> PipelineOutputs:
    raw_path = Path(raw_data_dir)
    deduplicated_dir = (
        Path(deduplicated_output_dir)
        if deduplicated_output_dir is not None
        else _default_stage_dir(raw_path, DEDUPLICATED_DIR_NAME)
    )
    cleaned_dir = (
        Path(cleaned_output_dir) if cleaned_output_dir is not None else _default_stage_dir(raw_path, CLEANED_DIR_NAME)
    )

    print("Step 1/2: Deduplicating data")
    deduplicated_file = deduplicate_data(raw_path, deduplicated_dir)

    print("Step 2/2: Cleaning data")
    cleaned_file = clean_data(deduplicated_file, cleaned_dir)
    flagged_file = cleaned_dir / DEFAULT_FLAGGED_FILE_NAME

    print("Pipeline complete")
    print(f"Deduplicated file: {deduplicated_file}")
    print(f"Cleaned file: {cleaned_file}")
    print(f"Rejected or flagged file: {flagged_file}")

    return PipelineOutputs(
        deduplicated_file=deduplicated_file,
        cleaned_file=cleaned_file,
        flagged_file=flagged_file,
    )


def _default_stage_dir(raw_data_dir: Path, stage_dir_name: str) -> Path:
    return raw_data_dir.parent / stage_dir_name if raw_data_dir.name == "raw" else raw_data_dir / stage_dir_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the comment data pipeline.")
    parser.add_argument("--raw-data-dir", default=str(DEFAULT_RAW_DATA_DIR), help="Folder containing source data files.")
    parser.add_argument(
        "--deduplicated-output-dir",
        default=None,
        help="Folder for deduplicated output. Defaults to a deduplicated folder next to raw data.",
    )
    parser.add_argument(
        "--cleaned-output-dir",
        default=None,
        help="Folder for cleaned output. Defaults to a cleaned folder next to raw data.",
    )
    args = parser.parse_args()

    run_pipeline(args.raw_data_dir, args.deduplicated_output_dir, args.cleaned_output_dir)


if __name__ == "__main__":
    main()
