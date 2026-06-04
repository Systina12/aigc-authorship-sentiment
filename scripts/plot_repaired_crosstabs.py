from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_repaired_analysis import DEFAULT_OUTPUT_DIR, write_crosstab_heatmap_figures


DEFAULT_TABLES_DIR = DEFAULT_OUTPUT_DIR / "tables"
DEFAULT_FIGURES_DIR = DEFAULT_OUTPUT_DIR / "figures"


def build_repaired_crosstab_charts(
    *,
    tables_dir: str | Path = DEFAULT_TABLES_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    top_n: int = 20,
) -> dict[str, Path]:
    if top_n < 1:
        raise ValueError("top_n must be greater than or equal to 1")
    return write_crosstab_heatmap_figures(
        tables_dir=Path(tables_dir),
        figures_dir=Path(figures_dir),
        top_n=top_n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot repaired-analysis crosstab charts from existing CSV tables.")
    parser.add_argument("--tables-dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    figures = build_repaired_crosstab_charts(
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        top_n=args.top_n,
    )
    for name, path in figures.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
