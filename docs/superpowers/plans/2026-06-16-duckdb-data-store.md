# DuckDB Data Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the predictor's data from CSV files into a single git-tracked DuckDB database that holds matches, predictions (committed + latest), goalscorers, and shootouts.

**Architecture:** A single new module `wc2026/db.py` owns every DuckDB interaction (connect, schema, deterministic `match_id`, read/write helpers, scoring view). `wc2026/data.py`'s `load_results()` is repointed at the DB while keeping its return contract identical, so the model, `build_training_frame`, `upcoming_fixtures`, and the existing test suite are untouched. A one-time `scripts/migrate_to_duckdb.py` seeds the DB from the current CSVs.

**Tech Stack:** Python 3.12, `uv`, DuckDB (embedded), pandas, pytest. Spec: `docs/superpowers/specs/2026-06-16-duckdb-data-store-design.md`.

**Convention:** Every commit message ends with the trailer
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` (omitted from the commands below for brevity — add it to each).

**Implementation refinement vs spec:** `match_id` is a documented *join key*, declared as a plain `TEXT` column (no enforced foreign key). DuckDB enforces declared FKs on insert, and historical goalscorer/shootout rows can reference matches outside the corpus; a plain join column keeps the migration robust. The relationships in the spec hold logically.

---

### Task 1: Add the DuckDB dependency

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via `uv add`)

- [ ] **Step 1: Add duckdb**

Run: `uv add duckdb`
Expected: resolves and installs duckdb; `pyproject.toml` gains a `duckdb` entry under dependencies.

- [ ] **Step 2: Verify it imports**

Run: `uv run python -c "import duckdb; print(duckdb.__version__)"`
Expected: prints a version (e.g. `1.x.y`), exit 0.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add duckdb dependency"
```

---

### Task 2: Deterministic `match_id`

**Files:**
- Create: `wc2026/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
import pandas as pd
import pytest
from wc2026 import db


def test_make_match_id_is_deterministic_slug():
    a = db.make_match_id("2026-06-16", "France", "Senegal")
    b = db.make_match_id(pd.Timestamp("2026-06-16"), "France", "Senegal")
    assert a == b == "20260616-france-senegal"


def test_make_match_id_slugifies_spaces_and_case():
    mid = db.make_match_id("2026-06-24", "Bosnia and Herzegovina", "Qatar")
    assert mid == "20260624-bosnia-and-herzegovina-qatar"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_db.py -q`
Expected: FAIL with `AttributeError: module 'wc2026.db' has no attribute 'make_match_id'` (or ImportError if file absent).

- [ ] **Step 3: Write minimal implementation**

```python
# wc2026/db.py
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = _DATA_DIR / "wc2026.duckdb"


def _slug(s: str) -> str:
    s = str(s).strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def make_match_id(match_date, home: str, away: str) -> str:
    """Deterministic, rebuild-stable id: YYYYMMDD-home-away (slugified).

    NOT an autoincrement: a DB rebuild must reproduce the same ids or the
    predictions <-> matches join breaks."""
    d = pd.Timestamp(match_date).strftime("%Y%m%d")
    return f"{d}-{_slug(home)}-{_slug(away)}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_db.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add wc2026/db.py tests/test_db.py
git commit -m "Add deterministic make_match_id"
```

---

### Task 3: Connection + schema (tables)

**Files:**
- Modify: `wc2026/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py  (append)
@pytest.fixture
def con():
    c = db.connect(":memory:")
    db.init_schema(c)
    yield c
    c.close()


def test_init_schema_creates_tables(con):
    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"matches", "predictions", "goalscorers", "shootouts"} <= names


def test_init_schema_is_idempotent(con):
    db.init_schema(con)  # second call must not raise
    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert "matches" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_db.py -q`
Expected: FAIL with `AttributeError: module 'wc2026.db' has no attribute 'connect'`.

- [ ] **Step 3: Write minimal implementation**

```python
# wc2026/db.py  (append)
def connect(path=DB_PATH, read_only: bool = False):
    return duckdb.connect(str(path), read_only=read_only)


_TABLES = [
    """CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY,
        date DATE,
        home_team TEXT, away_team TEXT,
        home_score INTEGER, away_score INTEGER,
        tournament TEXT,
        neutral BOOLEAN,
        city TEXT, country TEXT,
        stage TEXT, round TEXT, group_label TEXT,
        source TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS predictions (
        match_id TEXT,
        kind TEXT,
        pred_home_goals INTEGER, pred_away_goals INTEGER,
        outcome TEXT,
        lam_h DOUBLE, lam_a DOUBLE,
        p_result DOUBLE, p_home_g DOUBLE, p_away_g DOUBLE, p_gd DOUBLE, ep DOUBLE,
        model_as_of DATE,
        forecast_ts TIMESTAMP,
        PRIMARY KEY (match_id, kind)
    )""",
    """CREATE TABLE IF NOT EXISTS goalscorers (
        match_id TEXT,
        team TEXT, scorer TEXT, minute INTEGER, own_goal BOOLEAN, penalty BOOLEAN
    )""",
    """CREATE TABLE IF NOT EXISTS shootouts (
        match_id TEXT,
        winner TEXT, first_shooter TEXT
    )""",
]


def init_schema(con) -> None:
    for stmt in _TABLES:
        con.execute(stmt)
    _create_report_view(con)


def _create_report_view(con) -> None:
    # Placeholder until Task 4 fills in the scoring view.
    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_db.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add wc2026/db.py tests/test_db.py
git commit -m "Add DuckDB connect + schema init"
```

---

### Task 4: `v_model_report` scoring view

**Files:**
- Modify: `wc2026/db.py` (`_create_report_view`)
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py  (append)
from wc2026 import ScoringConfig, score_prediction


def _seed_match(con, mid, h, a, hs, as_):
    con.execute(
        "INSERT INTO matches (match_id, date, home_team, away_team, "
        "home_score, away_score, source) VALUES (?, DATE '2026-06-16', ?, ?, ?, ?, 'wc2026')",
        [mid, h, a, hs, as_],
    )


def _seed_committed(con, mid, ph, pa):
    con.execute(
        "INSERT INTO predictions (match_id, kind, pred_home_goals, "
        "pred_away_goals, outcome) VALUES (?, 'committed', ?, ?, 'H')",
        [mid, ph, pa],
    )


def test_report_view_points_match_score_prediction(con):
    cfg = ScoringConfig()
    cases = [("m1", 2, 1, 2, 1), ("m2", 1, 0, 0, 2), ("m3", 1, 1, 2, 0)]
    for mid, ph, pa, ah, ay in cases:
        _seed_match(con, mid, "H", "A", ah, ay)
        _seed_committed(con, mid, ph, pa)
    rows = con.execute(
        "SELECT match_id, points FROM v_model_report ORDER BY match_id"
    ).fetchall()
    got = {mid: pts for mid, pts in rows}
    for mid, ph, pa, ah, ay in cases:
        assert got[mid] == score_prediction(ph, pa, ah, ay, cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_db.py::test_report_view_points_match_score_prediction -q`
Expected: FAIL — `Catalog Error: Table with name v_model_report does not exist`.

- [ ] **Step 3: Implement the view**

Replace the placeholder `_create_report_view` body:

```python
# wc2026/db.py
def _create_report_view(con) -> None:
    # Office-pool scoring: a=3 correct outcome, b=1 per correct team goal
    # count, c=1 correct signed goal difference. Mirrors predict.score_prediction.
    con.execute("""
        CREATE OR REPLACE VIEW v_model_report AS
        SELECT
            m.match_id, m.date, m.home_team, m.away_team,
            m.home_score AS actual_h, m.away_score AS actual_a,
            p.pred_home_goals AS pred_h, p.pred_away_goals AS pred_a,
            (sign(p.pred_home_goals - p.pred_away_goals)
               = sign(m.home_score - m.away_score))::INT      AS outcome_ok,
            (p.pred_home_goals = m.home_score)::INT
               + (p.pred_away_goals = m.away_score)::INT       AS side_goals,
            ((p.pred_home_goals - p.pred_away_goals)
               = (m.home_score - m.away_score))::INT           AS gd_ok,
            (p.pred_home_goals = m.home_score
               AND p.pred_away_goals = m.away_score)::INT      AS exact_ok,
            ( 3 * (sign(p.pred_home_goals - p.pred_away_goals)
                     = sign(m.home_score - m.away_score))::INT
            + 1 * (p.pred_home_goals = m.home_score)::INT
            + 1 * (p.pred_away_goals = m.away_score)::INT
            + 1 * ((p.pred_home_goals - p.pred_away_goals)
                     = (m.home_score - m.away_score))::INT )    AS points
        FROM matches m
        JOIN predictions p
          ON p.match_id = m.match_id AND p.kind = 'committed'
        WHERE m.home_score IS NOT NULL AND m.away_score IS NOT NULL
    """)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_db.py -q`
Expected: all passed (5).

- [ ] **Step 5: Commit**

```bash
git add wc2026/db.py tests/test_db.py
git commit -m "Add v_model_report scoring view"
```

---

### Task 5: `load_matches` read helper

**Files:**
- Modify: `wc2026/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py  (append)
def test_load_matches_returns_loader_contract(con):
    con.execute(
        "INSERT INTO matches (match_id, date, home_team, away_team, home_score, "
        "away_score, tournament, neutral, city, country, source) VALUES "
        "('m1', DATE '2022-06-01', 'Spain', 'Italy', 2, 1, 'Friendly', TRUE, 'X', 'Y', 'upstream'),"
        "('m2', DATE '2026-06-16', 'France', 'Senegal', NULL, NULL, 'FIFA World Cup', TRUE, 'Z', 'W', 'wc2026')"
    )
    df = db.load_matches(con)
    assert list(df.columns) == [
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "city", "country", "neutral",
    ]
    assert str(df["date"].dtype).startswith("datetime64")
    assert df["neutral"].dtype == bool
    # unplayed match has NaN score (mirrors the CSV na_values behaviour)
    assert df.set_index("home_team").loc["France", "home_score"] != \
        df.set_index("home_team").loc["France", "home_score"]  # NaN != NaN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_db.py::test_load_matches_returns_loader_contract -q`
Expected: FAIL — `module 'wc2026.db' has no attribute 'load_matches'`.

- [ ] **Step 3: Write minimal implementation**

```python
# wc2026/db.py  (append)
_MATCH_COLS = ["date", "home_team", "away_team", "home_score", "away_score",
               "tournament", "city", "country", "neutral"]


def load_matches(con) -> pd.DataFrame:
    """All matches (played + unplayed), in the exact column shape that
    data.load_results historically returned from the CSV."""
    df = con.execute(
        f"SELECT {', '.join(_MATCH_COLS)} FROM matches ORDER BY date"
    ).df()
    df["date"] = pd.to_datetime(df["date"])
    # scores as float64 with NaN for unplayed, mirroring read_csv(na_values=["NA"])
    df["home_score"] = df["home_score"].astype("float64")
    df["away_score"] = df["away_score"].astype("float64")
    df["neutral"] = df["neutral"].astype(bool)
    for c in ["home_team", "away_team", "tournament", "city", "country"]:
        df[c] = df[c].astype("string")
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_db.py -q`
Expected: all passed (6).

- [ ] **Step 5: Commit**

```bash
git add wc2026/db.py tests/test_db.py
git commit -m "Add load_matches read helper"
```

---

### Task 6: Prediction writes (latest upsert + commit immutability)

**Files:**
- Modify: `wc2026/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_db.py  (append)
def _pred_df(mid, ph, pa, outcome="H"):
    return pd.DataFrame([{
        "match_id": mid, "pred_home_goals": ph, "pred_away_goals": pa,
        "outcome": outcome, "lam_h": 1.5, "lam_a": 0.8, "p_result": 0.5,
        "p_home_g": 0.3, "p_away_g": 0.4, "p_gd": 0.2, "ep": 2.5,
    }])


def test_upsert_latest_overwrites_in_place(con):
    db.upsert_latest_predictions(con, _pred_df("m1", 1, 0), "2026-06-10")
    db.upsert_latest_predictions(con, _pred_df("m1", 2, 1), "2026-06-15")
    rows = con.execute(
        "SELECT pred_home_goals, pred_away_goals FROM predictions "
        "WHERE match_id='m1' AND kind='latest'"
    ).fetchall()
    assert rows == [(2, 1)]  # single latest row, overwritten


def test_commit_snapshots_latest(con):
    db.upsert_latest_predictions(con, _pred_df("m1", 2, 1), "2026-06-15")
    n = db.commit_predictions(con, ["m1"], now="2026-06-16 12:00:00")
    assert n == 1
    row = con.execute(
        "SELECT pred_home_goals, pred_away_goals FROM predictions "
        "WHERE match_id='m1' AND kind='committed'"
    ).fetchone()
    assert row == (2, 1)


def test_commit_is_immutable_without_force(con):
    db.upsert_latest_predictions(con, _pred_df("m1", 2, 1), "2026-06-15")
    db.commit_predictions(con, ["m1"], now="2026-06-16 12:00:00")
    # a later re-forecast changes latest, then a second commit attempt
    db.upsert_latest_predictions(con, _pred_df("m1", 0, 0), "2026-06-16")
    n = db.commit_predictions(con, ["m1"], now="2026-06-16 13:00:00")
    assert n == 0  # refused
    row = con.execute(
        "SELECT pred_home_goals, pred_away_goals FROM predictions "
        "WHERE match_id='m1' AND kind='committed'"
    ).fetchone()
    assert row == (2, 1)  # original committed pick preserved


def test_commit_force_overwrites(con):
    db.upsert_latest_predictions(con, _pred_df("m1", 2, 1), "2026-06-15")
    db.commit_predictions(con, ["m1"], now="2026-06-16 12:00:00")
    db.upsert_latest_predictions(con, _pred_df("m1", 0, 0), "2026-06-16")
    n = db.commit_predictions(con, ["m1"], force=True, now="2026-06-16 13:00:00")
    assert n == 1
    row = con.execute(
        "SELECT pred_home_goals, pred_away_goals FROM predictions "
        "WHERE match_id='m1' AND kind='committed'"
    ).fetchone()
    assert row == (0, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_db.py -k "upsert_latest or commit" -q`
Expected: FAIL — `module 'wc2026.db' has no attribute 'upsert_latest_predictions'`.

- [ ] **Step 3: Write minimal implementation**

```python
# wc2026/db.py  (append)
_PRED_COLS = ["match_id", "kind", "pred_home_goals", "pred_away_goals",
              "outcome", "lam_h", "lam_a", "p_result", "p_home_g",
              "p_away_g", "p_gd", "ep", "model_as_of", "forecast_ts"]


def upsert_latest_predictions(con, df: pd.DataFrame, model_as_of, now=None) -> None:
    """Write/replace the single kind='latest' row per match."""
    rows = df.copy()
    rows["kind"] = "latest"
    rows["model_as_of"] = pd.Timestamp(model_as_of)
    rows["forecast_ts"] = pd.Timestamp(now) if now is not None else pd.Timestamp.utcnow()
    rows = rows[_PRED_COLS]
    con.register("_preds", rows)
    con.execute(
        f"INSERT OR REPLACE INTO predictions ({', '.join(_PRED_COLS)}) "
        f"SELECT {', '.join(_PRED_COLS)} FROM _preds"
    )
    con.unregister("_preds")


def commit_predictions(con, match_ids, force: bool = False, now=None) -> int:
    """Snapshot current 'latest' rows to 'committed' for the given matches.
    Refuses to overwrite an existing committed row unless force=True.
    Returns the number of committed rows written."""
    ids = list(dict.fromkeys(match_ids))
    if not ids:
        return 0
    ts = pd.Timestamp(now) if now is not None else pd.Timestamp.utcnow()
    con.register("_ids", pd.DataFrame({"match_id": ids}))
    guard = "" if force else (
        "AND p.match_id NOT IN (SELECT match_id FROM predictions WHERE kind='committed')"
    )
    written = con.execute(
        f"""INSERT OR REPLACE INTO predictions
            (match_id, kind, pred_home_goals, pred_away_goals, outcome,
             lam_h, lam_a, p_result, p_home_g, p_away_g, p_gd, ep,
             model_as_of, forecast_ts)
            SELECT p.match_id, 'committed', p.pred_home_goals, p.pred_away_goals,
             p.outcome, p.lam_h, p.lam_a, p.p_result, p.p_home_g, p.p_away_g,
             p.p_gd, p.ep, p.model_as_of, ?
            FROM predictions p
            WHERE p.kind='latest'
              AND p.match_id IN (SELECT match_id FROM _ids) {guard}
            RETURNING 1""",
        [ts],
    ).fetchall()
    con.unregister("_ids")
    return len(written)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_db.py -q`
Expected: all passed (10).

- [ ] **Step 5: Commit**

```bash
git add wc2026/db.py tests/test_db.py
git commit -m "Add prediction upsert + commit-immutability helpers"
```

---

### Task 7: `upsert_results` (fetcher hook)

**Files:**
- Modify: `wc2026/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py  (append)
def test_upsert_results_fills_scores(con):
    con.execute(
        "INSERT INTO matches (match_id, date, home_team, away_team, source) "
        "VALUES ('m1', DATE '2026-06-16', 'France', 'Senegal', 'wc2026')"
    )
    db.upsert_results(con, [{"match_id": "m1", "home_score": 2, "away_score": 1}])
    row = con.execute(
        "SELECT home_score, away_score FROM matches WHERE match_id='m1'"
    ).fetchone()
    assert row == (2, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_db.py::test_upsert_results_fills_scores -q`
Expected: FAIL — `module 'wc2026.db' has no attribute 'upsert_results'`.

- [ ] **Step 3: Write minimal implementation**

```python
# wc2026/db.py  (append)
def upsert_results(con, rows) -> None:
    """Fill scores for existing matches (the fetcher's write hook). `rows` is a
    list of dicts or a DataFrame with match_id, home_score, away_score."""
    df = pd.DataFrame(rows)
    if df.empty:
        return
    con.register("_res", df)
    con.execute(
        "UPDATE matches AS m SET home_score = r.home_score, "
        "away_score = r.away_score FROM _res r WHERE m.match_id = r.match_id"
    )
    con.unregister("_res")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_db.py -q`
Expected: all passed (11).

- [ ] **Step 5: Commit**

```bash
git add wc2026/db.py tests/test_db.py
git commit -m "Add upsert_results fetcher hook"
```

---

### Task 8: Match-id assignment + group-matchday derivation

**Files:**
- Modify: `wc2026/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_db.py  (append)
def test_assign_match_ids():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-16", "2026-06-16"]),
        "home_team": ["France", "Iraq"], "away_team": ["Senegal", "Norway"],
    })
    out = db.assign_match_ids(df)
    assert out["match_id"].tolist() == [
        "20260616-france-senegal", "20260616-iraq-norway"]


def test_derive_group_matchday_counts_appearances():
    # 2 groups (ABCD, EFGH style) over 3 rounds; matchday = each team's Nth game
    rounds = [
        ("2026-06-11", "A", "B"), ("2026-06-11", "C", "D"),
        ("2026-06-15", "A", "C"), ("2026-06-15", "B", "D"),
        ("2026-06-19", "A", "D"), ("2026-06-19", "B", "C"),
    ]
    df = pd.DataFrame(
        [{"date": pd.Timestamp(d), "home_team": h, "away_team": a}
         for d, h, a in rounds]
    )
    md = db.derive_group_matchday(df)
    assert md.tolist() == [1, 1, 2, 2, 3, 3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_db.py -k "assign_match_ids or matchday" -q`
Expected: FAIL — attributes not defined.

- [ ] **Step 3: Write minimal implementation**

```python
# wc2026/db.py  (append)
def assign_match_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["match_id"] = [
        make_match_id(d, h, a)
        for d, h, a in zip(df["date"], df["home_team"], df["away_team"])
    ]
    return df


def derive_group_matchday(group_df: pd.DataFrame) -> pd.Series:
    """Matchday (1..3) per group game, via each team's Nth appearance in date
    order. Both teams in a round-robin game share the same appearance count.
    Returns a Series aligned to group_df's index."""
    ordered = group_df.sort_values("date")
    counts: dict[str, int] = {}
    md: dict = {}
    for idx, r in ordered.iterrows():
        n = max(counts.get(r["home_team"], 0), counts.get(r["away_team"], 0)) + 1
        counts[r["home_team"]] = n
        counts[r["away_team"]] = n
        md[idx] = n
    return pd.Series(md).reindex(group_df.index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_db.py -q`
Expected: all passed (13).

- [ ] **Step 5: Commit**

```bash
git add wc2026/db.py tests/test_db.py
git commit -m "Add match-id assignment + group-matchday derivation"
```

---

### Task 9: Migration script + build the database

**Files:**
- Create: `scripts/migrate_to_duckdb.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test (build_matches transform)**

```python
# tests/test_db.py  (append)
from scripts import migrate_to_duckdb as mig


def test_build_matches_labels_wc_rows(tmp_path):
    results = tmp_path / "results.csv"
    wc = tmp_path / "wc.csv"
    hdr = "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
    results.write_text(
        hdr + "2024-01-01,Spain,Italy,2,1,Friendly,X,Y,FALSE\n"
    )
    wc.write_text(
        hdr
        + "2026-06-11,France,Senegal,1,0,FIFA World Cup,A,US,TRUE\n"
        + "2026-06-15,France,Iraq,NA,NA,FIFA World Cup,B,US,TRUE\n"
    )
    df = mig.build_matches(results, wc)
    by_id = df.set_index("match_id")
    assert by_id.loc["20240101-spain-italy", "source"] == "upstream"
    assert by_id.loc["20240101-spain-italy", "stage"] is None or \
        pd.isna(by_id.loc["20240101-spain-italy", "stage"])
    assert by_id.loc["20260611-france-senegal", "stage"] == "group"
    assert by_id.loc["20260611-france-senegal", "round"] == "MD1"
    assert by_id.loc["20260615-france-iraq", "round"] == "MD2"
    assert pd.isna(by_id.loc["20260611-france-senegal", "group_label"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_db.py::test_build_matches_labels_wc_rows -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.migrate_to_duckdb'`.

- [ ] **Step 3: Write the migration script**

```python
# scripts/migrate_to_duckdb.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_db.py -q`
Expected: all passed (14).

- [ ] **Step 5: Build the real database**

Run: `uv run python -m scripts.migrate_to_duckdb`
Expected: prints `seeded <N> matches (72 WC) -> wc2026.duckdb` where N is ~49.4k. File `data/wc2026.duckdb` now exists.

- [ ] **Step 6: Sanity-check the build**

Run:
```bash
uv run python -c "
from wc2026 import db
c = db.connect(db.DB_PATH, read_only=True)
print('WC group MD1 games:', c.execute(\"SELECT count(*) FROM matches WHERE stage='group' AND round='MD1'\").fetchone()[0])
print('distinct WC teams:', c.execute(\"SELECT count(DISTINCT t) FROM (SELECT home_team t FROM matches WHERE source='wc2026' UNION SELECT away_team FROM matches WHERE source='wc2026')\").fetchone()[0])
"
```
Expected: `WC group MD1 games: 24` (12 groups × 2 games each per matchday) and `distinct WC teams: 48`. If MD1 ≠ 24 or teams ≠ 48, the matchday derivation or the seed is wrong, stop and debug before committing.

- [ ] **Step 7: Commit (including the database binary)**

```bash
git add scripts/migrate_to_duckdb.py tests/test_db.py data/wc2026.duckdb
git commit -m "Add DuckDB migration and seed the database"
```

---

### Task 10: Repoint `data.load_results` at the database

**Files:**
- Modify: `wc2026/data.py:48-86` (the `load_results` function region)
- Test: `tests/test_db.py`, full suite

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py  (append)
from wc2026 import data as wcdata


def test_load_results_reads_from_db(tmp_path, monkeypatch):
    dbfile = tmp_path / "t.duckdb"
    c = db.connect(dbfile)
    db.init_schema(c)
    c.execute(
        "INSERT INTO matches (match_id, date, home_team, away_team, home_score, "
        "away_score, tournament, neutral, city, country, source) VALUES "
        "('m1', DATE '2022-06-01', 'Spain', 'Italy', 2, 1, 'Friendly', TRUE, 'X', 'Y', 'upstream')"
    )
    c.close()
    monkeypatch.setattr(db, "DB_PATH", dbfile)  # load_results reads db.DB_PATH at call time
    df = wcdata.load_results()
    assert {"home_team", "away_team", "home_score", "neutral"} <= set(df.columns)
    assert df.iloc[0]["home_team"] == "Spain"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_db.py::test_load_results_reads_from_db -q`
Expected: FAIL — the pre-change `load_results` reads `results.csv` and ignores `db.DB_PATH`, so `df.iloc[0]["home_team"]` is the first historical CSV match, not "Spain".

- [ ] **Step 3: Rewrite `load_results` to read from the DB**

In `wc2026/data.py`, replace the current `load_results` body (the function that reads `results.csv` + unions `wc-2026-games.csv`) with a DB-backed version. Keep `_read_results_csv` (the migration still uses it). Add the import at the top of the file: `from . import db`.

```python
# wc2026/data.py  (replace the load_results function)
def load_results(path: Path | str | None = None,
                 wc_games_path: Path | str | None = None) -> pd.DataFrame:
    """Load all matches (played + unplayed) from the DuckDB store, in the same
    column shape the CSV loader used to return. The `path`/`wc_games_path`
    arguments are retained for backwards compatibility and ignored (the data
    now lives in the database; see scripts.migrate_to_duckdb)."""
    con = db.connect(db.DB_PATH, read_only=True)
    try:
        return db.load_matches(con)
    finally:
        con.close()
```

- [ ] **Step 4: Run the targeted test**

Run: `uv run python -m pytest tests/test_db.py::test_load_results_reads_from_db -q`
Expected: PASS.

- [ ] **Step 5: Run the FULL suite (contract must hold)**

Run: `uv run python -m pytest -q`
Expected: all passed (existing 25 + the new db tests). If any model/predict test fails, the `load_matches` dtype contract diverged — fix `load_matches`, not the tests.

- [ ] **Step 6: End-to-end check the predictor still runs**

Run: `uv run python -m scripts.run_schedule --cached`
Expected: prints the 56-fixture prediction table with zero skips (reads matches from the DB now).
> If `--cached` fails because `model_cache.json` is absent, run without `--cached` (warm/cold fit) once.

- [ ] **Step 7: Commit**

```bash
git add wc2026/data.py tests/test_db.py
git commit -m "Read matches from DuckDB in load_results"
```

---

### Task 11: `run_schedule` writes latest predictions to the DB

**Files:**
- Modify: `scripts/run_schedule.py`

- [ ] **Step 1: Write the failing test (prediction-frame builder)**

Add a small pure helper to `run_schedule` so the DB-write payload is testable without a model fit.

```python
# tests/test_db.py  (append)
from scripts import run_schedule as rs


def test_predictions_frame_shape():
    rows = [{
        "date": "2026-06-16", "home": "France", "away": "Senegal",
        "pick": "1-0", "lam_h": 1.5, "lam_a": 1.1, "result": "H",
        "P_result": 0.46, "P_home_g": 0.33, "P_away_g": 0.34, "P_gd": 0.23,
        "EP": 2.29,
    }]
    df = rs.predictions_frame(rows)
    assert df.loc[0, "match_id"] == "20260616-france-senegal"
    assert df.loc[0, "pred_home_goals"] == 1
    assert df.loc[0, "pred_away_goals"] == 0
    assert df.loc[0, "outcome"] == "H"
    assert set(["lam_h", "lam_a", "p_result", "p_home_g", "p_away_g",
                "p_gd", "ep"]).issubset(df.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_db.py::test_predictions_frame_shape -q`
Expected: FAIL — `module 'scripts.run_schedule' has no attribute 'predictions_frame'`.

- [ ] **Step 3: Add `predictions_frame` and wire the DB write**

Add the import `from wc2026 import db` near the top of `scripts/run_schedule.py`, then add this helper:

```python
# scripts/run_schedule.py  (add near the other helpers)
def predictions_frame(rows: list[dict]) -> pd.DataFrame:
    """Map run_schedule's output rows to the predictions-table payload."""
    out = []
    for r in rows:
        ph, pa = (int(x) for x in r["pick"].split("-"))
        out.append({
            "match_id": db.make_match_id(r["date"], r["home"], r["away"]),
            "pred_home_goals": ph, "pred_away_goals": pa,
            "outcome": r["result"],
            "lam_h": r["lam_h"], "lam_a": r["lam_a"],
            "p_result": r["P_result"], "p_home_g": r["P_home_g"],
            "p_away_g": r["P_away_g"], "p_gd": r["P_gd"], "ep": r["EP"],
        })
    return pd.DataFrame(out)
```

Then, in `main()`, after `out.to_csv(OUT_PATH, index=False)`, persist the latest predictions:

```python
# scripts/run_schedule.py  (in main, after writing predictions.csv)
con = db.connect(db.DB_PATH)
try:
    db.upsert_latest_predictions(con, predictions_frame(rows), pre.as_of)
finally:
    con.close()
print(f"wrote {len(rows)} latest predictions to {db.DB_PATH.name}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_db.py::test_predictions_frame_shape -q`
Expected: PASS.

- [ ] **Step 5: End-to-end: run the predictor and confirm DB rows**

Run: `uv run python -m scripts.run_schedule --cached`
Then:
```bash
uv run python -c "
from wc2026 import db
c = db.connect(db.DB_PATH, read_only=True)
print('latest rows:', c.execute(\"SELECT count(*) FROM predictions WHERE kind='latest'\").fetchone()[0])
"
```
Expected: `latest rows: 56`.

- [ ] **Step 6: Commit (including the updated database)**

```bash
git add scripts/run_schedule.py tests/test_db.py data/wc2026.duckdb
git commit -m "Persist latest predictions to DuckDB from run_schedule"
```

---

### Task 12: `commit_picks.py` (the 90-minute lock)

**Files:**
- Create: `scripts/commit_picks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py  (append)
from scripts import commit_picks


def test_match_ids_for_date_filters_unplayed(con):
    con.execute(
        "INSERT INTO matches (match_id, date, home_team, away_team, source) VALUES "
        "('20260616-france-senegal', DATE '2026-06-16', 'France', 'Senegal', 'wc2026'),"
        "('20260617-england-croatia', DATE '2026-06-17', 'England', 'Croatia', 'wc2026')"
    )
    ids = commit_picks.match_ids_for_date(con, "2026-06-16")
    assert ids == ["20260616-france-senegal"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_db.py::test_match_ids_for_date_filters_unplayed -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.commit_picks'`.

- [ ] **Step 3: Write the script**

```python
# scripts/commit_picks.py
"""Lock in (commit) the current latest picks for matches on a given date.

Run on matchday (the user commits ~90 min before kickoff). Committed picks are
immutable unless --force is passed.

    uv run python -m scripts.commit_picks [YYYY-MM-DD] [--force]
"""
from __future__ import annotations

import sys
from datetime import date

from wc2026 import db


def match_ids_for_date(con, on_date) -> list[str]:
    rows = con.execute(
        "SELECT match_id FROM matches WHERE date = ? ORDER BY match_id",
        [str(on_date)],
    ).fetchall()
    return [r[0] for r in rows]


def main(on_date: str, force: bool = False) -> None:
    con = db.connect(db.DB_PATH)
    try:
        ids = match_ids_for_date(con, on_date)
        n = db.commit_predictions(con, ids, force=force)
        print(f"committed {n}/{len(ids)} picks for {on_date}"
              + (" (forced)" if force else ""))
    finally:
        con.close()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--force"]
    on = args[0] if args else date.today().isoformat()
    main(on, force="--force" in sys.argv[1:])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_db.py -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/commit_picks.py tests/test_db.py
git commit -m "Add commit_picks script (committed-pick lock)"
```

---

### Task 13: Documentation touch-up

**Files:**
- Modify: `CLAUDE.md`, `README.md`

- [ ] **Step 1: Update `CLAUDE.md`**

Add a short note under the data/environment section recording the new convention (no code, just the non-obvious facts):
- Data now lives in `data/wc2026.duckdb` (git-tracked), seeded by `scripts/migrate_to_duckdb.py` from the CSVs.
- `results.csv` is the upstream historical seed; `wc-2026-games.csv` was the one-time WC seed; both are import sources, not edited by hand.
- `wc2026/db.py` owns all DB access; `load_results()` reads matches from the DB.
- Predictions are stored as `committed` (locked ~90 min pre-kickoff via `scripts.commit_picks`) and `latest` (overwritten each run); `v_model_report` joins committed picks to actual results for model evaluation.

- [ ] **Step 2: Update `README.md`**

Add a "Data store" subsection mentioning the DuckDB file, the migration command, and the `run_schedule` / `commit_picks` workflow.

- [ ] **Step 3: Run the full suite one last time**

Run: `uv run python -m pytest -q`
Expected: all passed.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "Document the DuckDB data store"
```

---

## Notes for the implementer

- **DuckDB syntax used:** `INSERT OR REPLACE`, `RETURNING`, `UPDATE ... FROM`, `con.register`/`unregister` for zero-copy pandas, `(bool)::INT` casts, `sign()`. All supported by current DuckDB.
- **The slow model fit is not on this plan's critical path.** Only `run_schedule` triggers a fit, and the end-to-end steps use `--cached`. Never cold-fit in a loop.
- **The database binary is committed** at Tasks 9 and 11 (it's the source of truth). Keep an eye on its size; the data is small so it should stay a few MB.
- **`model_cache.json` stays a JSON file**, gitignored as before. This plan does not move it.
