from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion.parser import parse_cricsheet_csvs
from src.ingestion.pipeline import normalize_deliveries


def _write_sample_csv(path: Path) -> None:
    # Minimal mixed-format Cricsheet-style CSV with one ball and venue/date info.
    # Note: exact column counts vary in real files; parser should handle this gracefully.
    path.write_text(
        "\n".join(
            [
                "version,1.6.0",
                "info,date,2017/04/05",
                'info,venue,"Some Stadium"',
                "info,method,DLS",
                "ball,1,0.1,TeamA,BatterA,BatterB,BowlerA,1,0,,,,,,"",\"\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_parser_returns_dataframe(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_sample_csv(raw_dir / "123.csv")

    df = parse_cricsheet_csvs(raw_dir)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1


def test_normalized_required_columns_exist(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_sample_csv(raw_dir / "123.csv")

    raw_df = parse_cricsheet_csvs(raw_dir)
    deliveries = normalize_deliveries(raw_df)

    required = {
        "match_id",
        "innings",
        "over",
        "ball",
        "batsman",
        "bowler",
        "runs",
        "wicket",
        "venue",
        "date",
        "is_dls",
        "is_super_over",
        "is_abandoned",
    }
    assert required.issubset(set(deliveries.columns))
    assert bool(deliveries.loc[0, "is_dls"]) is True

