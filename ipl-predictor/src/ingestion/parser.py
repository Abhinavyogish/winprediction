"""Cricsheet IPL CSV ingestion.

Cricsheet CSVs contain mixed row types:
- `info,<key>,<value...>` metadata rows
- `ball,<innings>,<over.ball>,<batting_team>,<striker>,<non_striker>,<bowler>,<runs_off_bat>,<extras>,...`

This module parses all match CSVs in a directory into a single delivery-level DataFrame,
preserving edge cases like DLS, super overs, and abandoned/no-ball matches.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


_DLS_PATTERN = re.compile(r"\b(D/L|DLS|Duckworth|Lewis|Stern)\b", re.IGNORECASE)


@dataclass
class MatchInfo:
    """Holds match-level metadata discovered in `info` rows."""

    venue: Optional[str] = None
    date: Optional[str] = None  # keep raw string; pipeline can cast if needed
    method: Optional[str] = None
    is_dls: bool = False
    # additional info captured for debugging/traceability (not required downstream)
    extra: Dict[str, List[str]] = field(default_factory=dict)

    def observe(self, key: str, values: List[str]) -> None:
        key_norm = (key or "").strip().lower()
        value_joined = ",".join(values).strip()

        if key_norm == "venue" and value_joined:
            self.venue = value_joined
        elif key_norm == "date" and value_joined:
            self.date = value_joined
        elif key_norm == "method" and value_joined:
            self.method = value_joined

        # detect DLS from either key or value(s)
        if _DLS_PATTERN.search(key) or _DLS_PATTERN.search(value_joined):
            self.is_dls = True

        self.extra.setdefault(key_norm or "unknown", []).append(value_joined)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        s = str(value).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def _parse_over_ball(ob_str: str) -> Tuple[int, int]:
    """Parse an 'over.ball' string like '7.6' into (over, ball).

    Cricsheet uses additional deliveries in an over (wides/no-balls) producing
    values like 8.7, 8.8 etc. We keep the 'ball' component as an integer.
    """

    s = (ob_str or "").strip()
    if not s or "." not in s:
        return 0, 0
    over_s, ball_s = s.split(".", 1)
    return _safe_int(over_s, 0), _safe_int(ball_s, 0)


def _iter_csv_rows(path: Path) -> Iterable[List[str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            yield row


def parse_cricsheet_csv(path: Path) -> pd.DataFrame:
    """Parse a single Cricsheet match CSV into a delivery-level DataFrame.

    The returned DataFrame includes match-level columns repeated for each delivery:
    - match_id, venue, date, method
    - is_dls, is_super_over, is_abandoned

    Delivery-level extracted columns:
    - innings, over, ball
    - batsman, bowler
    - runs_off_bat, extras
    - dismissal_kind
    """

    match_id = path.stem
    info = MatchInfo()
    deliveries: List[Dict[str, Any]] = []
    saw_ball = False
    is_super_over = False

    try:
        for row in _iter_csv_rows(path):
            row_type = (row[0] or "").strip().lower()
            if row_type == "info":
                if len(row) >= 3:
                    key = row[1]
                    values = row[2:]
                    info.observe(key, values)
                continue

            if row_type != "ball":
                continue

            # Expected minimal columns (based on observed format):
            # 0 type, 1 innings, 2 over.ball, 3 batting_team, 4 striker, 5 non_striker,
            # 6 bowler, 7 runs_off_bat, 8 extras, ... , wicket_type (often near end)
            if len(row) < 9:
                logger.warning("Skipping short ball row in %s: %r", path.name, row)
                continue

            saw_ball = True
            innings = _safe_int(row[1], 0)
            if innings > 2:
                is_super_over = True

            over, ball = _parse_over_ball(row[2])
            batsman = (row[4] or "").strip()
            bowler = (row[6] or "").strip()
            runs_off_bat = _safe_int(row[7], 0)
            extras = _safe_int(row[8], 0)

            dismissal_kind = ""
            # In Cricsheet CSVs, wicket type commonly sits at index 14, and
            # player dismissed at index 15, but formats can vary across versions.
            # We prefer the last non-empty wicket-type-like field if present.
            # Strategy: look for a known dismissal kind by scanning tail fields.
            # If that's too strict, fall back to column 14 if available.
            tail = [c.strip() for c in row[9:] if isinstance(c, str)]
            # Common dismissal kinds in Cricsheet: caught, bowled, run out, lbw, stumped, hit wicket, retired hurt, obstructing the field, etc.
            # We cannot enumerate all reliably; instead, treat the first non-empty
            # token in the last ~6 columns as dismissal_kind when any wicket occurs.
            for token in reversed(tail[-8:]):
                if token:
                    dismissal_kind = token
                    break
            # Heuristic: if dismissal_kind looks like a player name (contains space and no lowercase dismissal keyword),
            # and we have a better candidate at fixed index 14, prefer fixed index.
            if len(row) > 14 and (dismissal_kind == "" or dismissal_kind == (row[-1] or "").strip()):
                candidate = (row[14] or "").strip()
                if candidate:
                    dismissal_kind = candidate

            deliveries.append(
                {
                    "match_id": match_id,
                    "innings": innings,
                    "over": over,
                    "ball": ball,
                    "batsman": batsman,
                    "bowler": bowler,
                    "runs_off_bat": runs_off_bat,
                    "extras": extras,
                    "dismissal_kind": dismissal_kind,
                    "venue": info.venue,
                    "date": info.date,
                    "method": info.method,
                    "is_dls": bool(info.is_dls),
                    "is_super_over": False,  # fill after loop
                    "is_abandoned": False,  # fill after loop
                }
            )
    except Exception:
        logger.exception("Failed parsing %s", path)

    if not saw_ball:
        # Abandoned/no-ball match: emit one placeholder row so match is preserved.
        deliveries.append(
            {
                "match_id": match_id,
                "innings": 0,
                "over": 0,
                "ball": 0,
                "batsman": "",
                "bowler": "",
                "runs_off_bat": 0,
                "extras": 0,
                "dismissal_kind": "",
                "venue": info.venue,
                "date": info.date,
                "method": info.method,
                "is_dls": bool(info.is_dls),
                "is_super_over": False,
                "is_abandoned": True,
            }
        )

    df = pd.DataFrame(deliveries)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if not df.empty:
        df["is_super_over"] = bool(is_super_over)
        if "is_abandoned" not in df.columns:
            df["is_abandoned"] = False
    return df


def parse_cricsheet_csvs(raw_dir: Path) -> pd.DataFrame:
    """Parse all `*.csv` in `raw_dir` into a single DataFrame.

    This function never drops matches due to edge cases; abandoned matches are
    represented by a placeholder row.
    """

    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    paths = sorted(raw_dir.glob("*.csv"))
    if not paths:
        logger.warning("No CSV files found under %s", raw_dir)
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for p in paths:
        frames.append(parse_cricsheet_csv(p))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

