from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FeaturePaths:
    match_features: Path
    innings_features: Path
    player_features: Path


MATCH_FEATURE_COLUMNS = [
    "match_id",
    "venue",
    "date",
    "is_dls",
    "is_super_over",
    "is_abandoned",
    "total_runs",
    "total_wickets",
    "balls",
    "overs",
    "run_rate",
]

INNINGS_FEATURE_COLUMNS = [
    "match_id",
    "innings",
    "runs",
    "wickets",
    "balls",
    "overs",
    "run_rate",
]

PLAYER_FEATURE_COLUMNS = [
    "player",
    "runs_scored",
    "balls_faced",
    "wickets_taken",
    "balls_bowled",
]


def _read_inputs(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(processed_dir)
    matches = pd.read_parquet(processed_dir / "matches.parquet")
    deliveries = pd.read_parquet(processed_dir / "deliveries.parquet")
    return matches, deliveries


def build_match_features(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    d = deliveries.copy()
    d["match_id"] = d["match_id"].astype(str)

    in_play = d["innings"].astype(int) > 0
    balls = d.loc[in_play].groupby("match_id")["ball"].size().rename("balls")
    totals = d.loc[in_play].groupby("match_id").agg(total_runs=("runs", "sum"), total_wickets=("wicket", "sum"))
    agg = totals.join(balls, how="outer").fillna(0)
    agg["balls"] = agg["balls"].astype(int)
    agg["overs"] = (agg["balls"] / 6.0).astype(float)
    agg["run_rate"] = (agg["total_runs"] / agg["overs"].where(agg["overs"] > 0, pd.NA)).fillna(0.0).astype(float)
    agg = agg.reset_index()

    m = matches.copy()
    m["match_id"] = m["match_id"].astype(str)
    out = m.merge(agg, on="match_id", how="left").fillna(
        {
            "total_runs": 0,
            "total_wickets": 0,
            "balls": 0,
            "overs": 0.0,
            "run_rate": 0.0,
        }
    )
    out["total_runs"] = out["total_runs"].astype(int)
    out["total_wickets"] = out["total_wickets"].astype(int)
    out["balls"] = out["balls"].astype(int)
    out["overs"] = out["overs"].astype(float)
    out["run_rate"] = out["run_rate"].astype(float)

    return out[MATCH_FEATURE_COLUMNS]


def build_innings_features(deliveries: pd.DataFrame) -> pd.DataFrame:
    d = deliveries.copy()
    d["match_id"] = d["match_id"].astype(str)
    d["innings"] = pd.to_numeric(d["innings"], errors="coerce").fillna(0).astype(int)

    in_play = d["innings"] > 0
    grouped = d.loc[in_play].groupby(["match_id", "innings"], as_index=False)
    out = grouped.agg(runs=("runs", "sum"), wickets=("wicket", "sum"), balls=("ball", "size"))
    out["runs"] = out["runs"].astype(int)
    out["wickets"] = out["wickets"].astype(int)
    out["balls"] = out["balls"].astype(int)
    out["overs"] = (out["balls"] / 6.0).astype(float)
    out["run_rate"] = (out["runs"] / out["overs"].where(out["overs"] > 0, pd.NA)).fillna(0.0).astype(float)
    return out[INNINGS_FEATURE_COLUMNS]


def build_player_features(deliveries: pd.DataFrame) -> pd.DataFrame:
    d = deliveries.copy()
    in_play = pd.to_numeric(d["innings"], errors="coerce").fillna(0).astype(int) > 0
    d = d.loc[in_play].copy()

    bat = (
        d.groupby("batsman", as_index=False)
        .agg(runs_scored=("runs", "sum"), balls_faced=("ball", "size"))
        .rename(columns={"batsman": "player"})
    )
    bat["runs_scored"] = bat["runs_scored"].astype(int)
    bat["balls_faced"] = bat["balls_faced"].astype(int)

    bowl = (
        d.groupby("bowler", as_index=False)
        .agg(wickets_taken=("wicket", "sum"), balls_bowled=("ball", "size"))
        .rename(columns={"bowler": "player"})
    )
    bowl["wickets_taken"] = bowl["wickets_taken"].astype(int)
    bowl["balls_bowled"] = bowl["balls_bowled"].astype(int)

    out = bat.merge(bowl, on="player", how="outer").fillna(0)
    out["player"] = out["player"].fillna("").astype(str)
    for col in ["runs_scored", "balls_faced", "wickets_taken", "balls_bowled"]:
        out[col] = out[col].astype(int)

    out = out.loc[out["player"].str.strip().ne("")].copy()
    return out[PLAYER_FEATURE_COLUMNS].sort_values("player").reset_index(drop=True)


def build_all_features(
    processed_dir: Path | None = None,
    output_dir: Path | None = None,
) -> FeaturePaths:
    root = project_root()
    processed_dir = Path(processed_dir) if processed_dir is not None else (root / "data" / "processed")
    output_dir = Path(output_dir) if output_dir is not None else processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    matches, deliveries = _read_inputs(processed_dir)

    match_features = build_match_features(matches, deliveries)
    innings_features = build_innings_features(deliveries)
    player_features = build_player_features(deliveries)

    paths = FeaturePaths(
        match_features=output_dir / "match_features.parquet",
        innings_features=output_dir / "innings_features.parquet",
        player_features=output_dir / "player_features.parquet",
    )

    match_features.to_parquet(paths.match_features, index=False)
    innings_features.to_parquet(paths.innings_features, index=False)
    player_features.to_parquet(paths.player_features, index=False)

    return paths


def main() -> None:
    paths = build_all_features()
    counts: Dict[str, int] = {
        "match_features": len(pd.read_parquet(paths.match_features)),
        "innings_features": len(pd.read_parquet(paths.innings_features)),
        "player_features": len(pd.read_parquet(paths.player_features)),
    }
    print(f"match_features.parquet rows: {counts['match_features']}")
    print(f"innings_features.parquet rows: {counts['innings_features']}")
    print(f"player_features.parquet rows: {counts['player_features']}")


if __name__ == "__main__":
    main()

