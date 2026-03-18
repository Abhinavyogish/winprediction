from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.build_features import FeaturePaths, build_all_features, project_root
from src.features.validate_features import validate_all_features


def test_build_and_validate_features(tmp_path: Path) -> None:
    root = project_root()
    processed_dir = root / "data" / "processed"

    paths = build_all_features(processed_dir=processed_dir, output_dir=tmp_path)
    assert isinstance(paths, FeaturePaths)

    assert paths.match_features.exists()
    assert paths.innings_features.exists()
    assert paths.player_features.exists()

    # Basic sanity checks
    mf = pd.read_parquet(paths.match_features)
    assert len(mf) > 0
    assert mf["match_id"].nunique() == len(mf)

    results = validate_all_features(paths)
    assert {k for k, v in results.items() if v.ok} == {"match_features", "innings_features", "player_features"}

