"""ETL pipeline: parse Cricsheet CSVs -> normalized parquet datasets.

Run from the project root:
    python -m src.ingestion.pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.ingestion.parser import parse_cricsheet_csvs

logger = logging.getLogger(__name__)


REQUIRED_DELIVERIES_COLUMNS = [
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
]


def project_root() -> Path:
    """Resolve the project root based on this file's location."""

    return Path(__file__).resolve().parents[2]


def normalize_deliveries(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize parsed deliveries into the unified schema."""

    if raw_df.empty:
        return pd.DataFrame(columns=REQUIRED_DELIVERIES_COLUMNS)

    df = raw_df.copy()
    df["runs"] = df.get("runs_off_bat", 0).fillna(0).astype(int) + df.get("extras", 0).fillna(0).astype(int)
    df["wicket"] = df.get("dismissal_kind", "").fillna("").astype(str).str.strip().ne("").astype(int)

    out = df[
        [
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
        ]
    ].copy()

    # Ensure types are stable for parquet
    out["match_id"] = out["match_id"].astype(str)
    for col in ["innings", "over", "ball", "runs", "wicket"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in ["is_dls", "is_super_over", "is_abandoned"]:
        out[col] = out[col].fillna(False).astype(bool)

    return out


def build_matches_table(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Create match-level table: match_id, venue, date, and flags."""

    if deliveries.empty:
        return pd.DataFrame(columns=["match_id", "venue", "date", "is_dls", "is_super_over", "is_abandoned"])

    def _first_non_null(s: pd.Series):
        s2 = s.dropna()
        for v in s2.tolist():
            if isinstance(v, str) and v.strip() == "":
                continue
            return v
        return None

    grouped = deliveries.groupby("match_id", as_index=False)
    matches = grouped.agg(
        venue=("venue", _first_non_null),
        date=("date", _first_non_null),
        is_dls=("is_dls", "max"),
        is_super_over=("is_super_over", "max"),
        is_abandoned=("is_abandoned", "max"),
    )
    return matches[["match_id", "venue", "date", "is_dls", "is_super_over", "is_abandoned"]]


def run_pipeline(raw_dir: Path, processed_dir: Path) -> Tuple[Path, Path]:
    """Run end-to-end pipeline and write parquet outputs.

    Returns:
        (matches_path, deliveries_path)
    """

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw CSVs from %s", raw_dir)
    raw_df = parse_cricsheet_csvs(raw_dir)
    logger.info("Parsed %d rows", len(raw_df))

    deliveries = normalize_deliveries(raw_df)
    matches = build_matches_table(deliveries)

    deliveries_path = processed_dir / "deliveries.parquet"
    matches_path = processed_dir / "matches.parquet"

    logger.info("Writing %s (%d rows)", deliveries_path, len(deliveries))
    deliveries.to_parquet(deliveries_path, index=False)

    logger.info("Writing %s (%d rows)", matches_path, len(matches))
    matches.to_parquet(matches_path, index=False)

    return matches_path, deliveries_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    root = project_root()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"

    if not raw_dir.exists() or not any(raw_dir.glob("*.csv")):
        logger.warning(
            "No CSVs found in %s. Copy/move your Cricsheet IPL CSVs there (e.g. from your existing workspace data/raw).",
            raw_dir,
        )

    run_pipeline(raw_dir=raw_dir, processed_dir=processed_dir)


if __name__ == "__main__":
    main()

