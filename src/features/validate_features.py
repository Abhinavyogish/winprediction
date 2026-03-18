from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.features.build_features import FeaturePaths, project_root


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: List[str]


def _expect_columns(df: pd.DataFrame, expected: List[str]) -> List[str]:
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    errors: List[str] = []
    if missing:
        errors.append(f"missing_columns={missing}")
    if extra:
        errors.append(f"extra_columns={extra}")
    return errors


def _expect_non_empty(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["table_is_empty"]
    return []


def validate_match_features(df: pd.DataFrame) -> ValidationResult:
    expected = [
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
    errors = []
    errors += _expect_columns(df, expected)
    errors += _expect_non_empty(df)
    return ValidationResult(ok=(len(errors) == 0), errors=errors)


def validate_innings_features(df: pd.DataFrame) -> ValidationResult:
    expected = ["match_id", "innings", "runs", "wickets", "balls", "overs", "run_rate"]
    errors = []
    errors += _expect_columns(df, expected)
    errors += _expect_non_empty(df)
    if "innings" in df.columns and (df["innings"].astype(int) <= 0).any():
        errors.append("innings_contains_non_positive")
    return ValidationResult(ok=(len(errors) == 0), errors=errors)


def validate_player_features(df: pd.DataFrame) -> ValidationResult:
    expected = ["player", "runs_scored", "balls_faced", "wickets_taken", "balls_bowled"]
    errors = []
    errors += _expect_columns(df, expected)
    errors += _expect_non_empty(df)
    if "player" in df.columns and df["player"].astype(str).str.strip().eq("").any():
        errors.append("player_contains_blank")
    return ValidationResult(ok=(len(errors) == 0), errors=errors)


def validate_all_features(paths: FeaturePaths) -> Dict[str, ValidationResult]:
    results: Dict[str, ValidationResult] = {}
    mf = pd.read_parquet(paths.match_features)
    inf = pd.read_parquet(paths.innings_features)
    pf = pd.read_parquet(paths.player_features)

    results["match_features"] = validate_match_features(mf)
    results["innings_features"] = validate_innings_features(inf)
    results["player_features"] = validate_player_features(pf)
    return results


def _print_result(name: str, res: ValidationResult) -> None:
    if res.ok:
        print(f"{name}: PASS")
    else:
        print(f"{name}: FAIL - " + "; ".join(res.errors))


def main() -> None:
    root = project_root()
    processed_dir = root / "data" / "processed"
    paths = FeaturePaths(
        match_features=processed_dir / "match_features.parquet",
        innings_features=processed_dir / "innings_features.parquet",
        player_features=processed_dir / "player_features.parquet",
    )

    results = validate_all_features(paths)
    for name, res in results.items():
        _print_result(name, res)


if __name__ == "__main__":
    main()

