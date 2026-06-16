"""One-time, idempotent seed of the DuckDB store from the CSV files.

    uv run python -m scripts.migrate_to_duckdb
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc2026 import db
from wc2026.data import (DEFAULT_RESULTS_PATH, DEFAULT_WC_GAMES_PATH,
                         _read_results_csv)

GOALSCORERS_PATH = db._DATA_DIR / "goalscorers.csv"
SHOOTOUTS_PATH = db._DATA_DIR / "shootouts.csv"

_MATCH_TABLE_COLS = ["match_id", "date", "home_team", "away_team", "home_score",
                     "away_score", "tournament", "neutral", "city", "country",
                     "stage", "round", "group_label", "source"]


def build_matches(results_csv: Path | str = DEFAULT_RESULTS_PATH,
                  wc_csv: Path | str = DEFAULT_WC_GAMES_PATH) -> pd.DataFrame:
    """Union historical + WC CSVs, assign match_id, label WC stage/round."""
    hist = _read_results_csv(results_csv)
    hist["source"] = "upstream"
    wc = _read_results_csv(wc_csv)
    wc["source"] = "wc2026"
    df = pd.concat([hist, wc], ignore_index=True)
    df = df.drop_duplicates(
        subset=["date", "home_team", "away_team"], keep="last"
    ).reset_index(drop=True)
    df = db.assign_match_ids(df)

    df["stage"] = pd.NA
    df["round"] = pd.NA
    df["group_label"] = pd.NA   # sourced later (see spec)

    wc_mask = df["source"].eq("wc2026")
    df.loc[wc_mask, "stage"] = "group"   # all current WC rows are group stage
    md = db.derive_group_matchday(df[wc_mask])
    df.loc[wc_mask, "round"] = md.map(lambda n: f"MD{int(n)}")
    return df[_MATCH_TABLE_COLS]


def _scorer_table(path: Path, cols: list[str]) -> pd.DataFrame:
    # Rows are NOT deduped: goalscorers/shootouts are seed-only (no read
    # consumers in the model path). Add dedup here if a feature joins them.
    raw = pd.read_csv(path, na_values=["NA"])
    raw["match_id"] = [
        db.make_match_id(d, h, a)
        for d, h, a in zip(raw["date"], raw["home_team"], raw["away_team"])
    ]
    return raw[["match_id", *cols]]


def main() -> None:
    con = db.connect()
    db.init_schema(con)
    for tbl in ("matches", "goalscorers", "shootouts"):
        con.execute(f"DELETE FROM {tbl}")

    matches = build_matches()
    con.register("_m", matches)
    con.execute(f"INSERT INTO matches ({', '.join(_MATCH_TABLE_COLS)}) "
                f"SELECT {', '.join(_MATCH_TABLE_COLS)} FROM _m")
    con.unregister("_m")

    gs = _scorer_table(GOALSCORERS_PATH,
                       ["team", "scorer", "minute", "own_goal", "penalty"])
    con.register("_gs", gs)
    con.execute("INSERT INTO goalscorers SELECT * FROM _gs")
    con.unregister("_gs")

    so = _scorer_table(SHOOTOUTS_PATH, ["winner", "first_shooter"])
    con.register("_so", so)
    con.execute("INSERT INTO shootouts SELECT * FROM _so")
    con.unregister("_so")

    n_matches = con.execute("SELECT count(*) FROM matches").fetchone()[0]
    n_wc = con.execute(
        "SELECT count(*) FROM matches WHERE source='wc2026'").fetchone()[0]
    con.close()
    print(f"seeded {n_matches} matches ({n_wc} WC) -> {db.DB_PATH.name}")


if __name__ == "__main__":
    main()
