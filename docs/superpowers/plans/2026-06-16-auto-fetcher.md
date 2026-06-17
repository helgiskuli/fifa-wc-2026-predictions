# Auto-fetcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pull match results from online sources into the DuckDB store via a pluggable provider interface, writing final scores and inserting new fixtures, so the predictor stays current without manual data entry.

**Architecture:** `wc2026/providers.py` defines a normalized `MatchRecord` and a `ResultsProvider` protocol, with a concrete `Martj42CsvProvider` (GitHub CSV feed) and a stubbed `LiveApiProvider`. `wc2026/fetch.py` holds provider-agnostic `reconcile()` (write policy + change report) calling a new `db.upsert_matches()`. `scripts/fetch_results.py` is the manual CLI. Providers are the only modules that touch the network; `db` stays policy-free.

**Tech Stack:** Python 3.12, `uv`, DuckDB, pandas, httpx, pytest. Spec: `docs/superpowers/specs/2026-06-16-auto-fetcher-design.md`.

**Convention:** Every commit message ends with the trailer
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` (add it to each commit below).

**Refinement vs spec:** `db.upsert_matches` conflict-update is **score-only** on existing rows (inserts new rows with full metadata; on `match_id` conflict updates only `home_score`/`away_score`). A full metadata upsert would clobber the seeded WC `stage`/`round`/`group_label`/`source` when martj42 backfills a played match (those records carry no WC labels). Fixture metadata-shift reconciliation is deferred to the live provider that needs it. Also: `MatchRecord` carries a `source` field (each provider stamps it) so `upsert_matches` needs no source argument.

---

## File Structure

- Create `wc2026/providers.py` — `MatchRecord`, `ResultsProvider` protocol, `Martj42CsvProvider`, `LiveApiProvider` (stub). Only network-touching module.
- Create `wc2026/fetch.py` — `ChangeReport`, `reconcile()`. No network, no raw SQL (delegates to `db`).
- Modify `wc2026/db.py` — add `upsert_matches()`.
- Create `scripts/fetch_results.py` — CLI (`--source`, `--corpus-refresh`, `--dry-run`).
- Test `tests/test_fetch.py` — providers, `reconcile`, CLI helpers.
- Test `tests/test_db.py` — `upsert_matches` (append).
- Modify `pyproject.toml` / `uv.lock` (httpx), `CLAUDE.md`, `README.md`, `.claude/commands/reforecast.md`.

---

### Task 1: Add the httpx dependency

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via `uv add`)

- [ ] **Step 1: Add httpx**

Run: `uv add httpx`
Expected: resolves and installs httpx; `pyproject.toml` gains an `httpx` entry under dependencies.

- [ ] **Step 2: Verify it imports**

Run: `uv run python -c "import httpx; print(httpx.__version__)"`
Expected: prints a version, exit 0.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add httpx dependency"
```

---

### Task 2: `MatchRecord` + `ResultsProvider` + live stub

**Files:**
- Create: `wc2026/providers.py`
- Test: `tests/test_fetch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetch.py
import pytest

from wc2026 import providers
from wc2026.providers import MatchRecord


def test_match_record_is_frozen_with_defaults():
    r = MatchRecord(date="2026-06-16", home_team="France", away_team="Senegal",
                    status="FINISHED", source="upstream",
                    home_score=1, away_score=0)
    assert r.home_team == "France"
    assert r.neutral is False  # default
    assert r.stage is None     # default
    with pytest.raises(Exception):
        r.home_score = 9  # frozen


def test_live_provider_fetch_not_implemented():
    p = providers.LiveApiProvider()
    assert p.name == "live"
    with pytest.raises(NotImplementedError):
        p.fetch()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_fetch.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'wc2026.providers'`.

- [ ] **Step 3: Write minimal implementation**

```python
# wc2026/providers.py
"""Result-source adapters. The ONLY modules that touch the network.

Each provider normalizes its source into MatchRecord objects in the DB's own
team-naming convention; the rest of the pipeline is source-agnostic.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import pandas as pd

MARTJ42_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)


@dataclass(frozen=True)
class MatchRecord:
    """One match from a provider, normalized to the DB's columns/naming.

    A scheduled (unplayed) fixture has status != 'FINISHED' and no scores."""
    date: str
    home_team: str
    away_team: str
    status: str          # 'FINISHED' | 'SCHEDULED' | 'IN_PLAY' | ...
    source: str          # 'upstream' (martj42) | provider tag (live)
    home_score: int | None = None
    away_score: int | None = None
    tournament: str | None = None
    neutral: bool = False
    city: str | None = None
    country: str | None = None
    stage: str | None = None
    round: str | None = None
    group_label: str | None = None


@runtime_checkable
class ResultsProvider(Protocol):
    name: str

    def fetch(self) -> list[MatchRecord]:
        ...


def _httpx_get(url: str) -> str:
    import httpx

    resp = httpx.get(url, timeout=30.0)
    resp.raise_for_status()
    return resp.text


class LiveApiProvider:
    """Fresh WC scores + emerging knockout fixtures. Concrete API chosen later
    (see docs/superpowers/specs/2026-06-16-auto-fetcher-design.md). Reads its
    key from WC_RESULTS_API_KEY; maps provider team names to the DB convention."""

    name = "live"

    def __init__(self) -> None:
        self._api_key = os.environ.get("WC_RESULTS_API_KEY")

    def fetch(self) -> list[MatchRecord]:
        raise NotImplementedError(
            "LiveApiProvider is not yet implemented: choose a concrete results "
            "API, set WC_RESULTS_API_KEY, and add a team-name map. See the "
            "auto-fetcher design spec."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_fetch.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add wc2026/providers.py tests/test_fetch.py
git commit -m "Add MatchRecord, ResultsProvider protocol, live-API stub"
```

---

### Task 3: `Martj42CsvProvider`

**Files:**
- Modify: `wc2026/providers.py`
- Test: `tests/test_fetch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetch.py  (append)
_SAMPLE_CSV = (
    "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
    "2026-06-11,France,Senegal,1,0,FIFA World Cup,Guadalajara,Mexico,TRUE\n"
    "2024-01-01,Spain,Italy,2,1,Friendly,Madrid,Spain,FALSE\n"
)


def test_martj42_provider_parses_csv():
    p = providers.Martj42CsvProvider(http_get=lambda url: _SAMPLE_CSV)
    recs = p.fetch()
    assert p.name == "martj42"
    assert len(recs) == 2
    fr = next(r for r in recs if r.home_team == "France")
    assert fr.home_score == 1 and fr.away_score == 0
    assert fr.status == "FINISHED"
    assert fr.neutral is True
    assert fr.source == "upstream"
    assert fr.tournament == "FIFA World Cup"
    assert fr.stage is None and fr.group_label is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_fetch.py::test_martj42_provider_parses_csv -q`
Expected: FAIL — `module 'wc2026.providers' has no attribute 'Martj42CsvProvider'`.

- [ ] **Step 3: Write minimal implementation**

```python
# wc2026/providers.py  (append)
class Martj42CsvProvider:
    """The martj42/international_results GitHub feed: the full historical corpus
    plus WC matches once played (lagged). Names already match the DB (the DB was
    seeded from this feed), so no name mapping. Played matches only."""

    name = "martj42"
    source = "upstream"

    def __init__(self, http_get: Callable[[str], str] | None = None,
                 url: str = MARTJ42_RESULTS_URL) -> None:
        self._get = http_get or _httpx_get
        self._url = url

    def fetch(self) -> list[MatchRecord]:
        df = pd.read_csv(io.StringIO(self._get(self._url)), na_values=["NA"])
        df["neutral"] = df["neutral"].astype("string").str.upper().eq("TRUE")
        out: list[MatchRecord] = []
        for r in df.itertuples(index=False):
            played = pd.notna(r.home_score) and pd.notna(r.away_score)
            out.append(MatchRecord(
                date=str(r.date),
                home_team=r.home_team,
                away_team=r.away_team,
                status="FINISHED" if played else "SCHEDULED",
                source=self.source,
                home_score=int(r.home_score) if played else None,
                away_score=int(r.away_score) if played else None,
                tournament=None if pd.isna(r.tournament) else r.tournament,
                neutral=bool(r.neutral),
                city=None if pd.isna(r.city) else r.city,
                country=None if pd.isna(r.country) else r.country,
            ))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_fetch.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add wc2026/providers.py tests/test_fetch.py
git commit -m "Add Martj42CsvProvider"
```

---

### Task 4: `db.upsert_matches`

**Files:**
- Modify: `wc2026/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py  (append; `con` fixture already exists in this file)
def test_upsert_matches_inserts_new_and_updates_score_only(con):
    # seed an existing WC fixture with labels and NULL score
    con.execute(
        "INSERT INTO matches (match_id, date, home_team, away_team, tournament, "
        "neutral, stage, round, group_label, source) VALUES "
        "('20260611-france-senegal', DATE '2026-06-11', 'France', 'Senegal', "
        "'FIFA World Cup', TRUE, 'group', 'MD1', 'A', 'wc2026')"
    )
    rows = [
        # existing match: fill score; carries no labels + source 'upstream'
        {"match_id": "20260611-france-senegal", "date": "2026-06-11",
         "home_team": "France", "away_team": "Senegal", "home_score": 1,
         "away_score": 0, "tournament": "FIFA World Cup", "neutral": True,
         "city": None, "country": None, "stage": None, "round": None,
         "group_label": None, "source": "upstream"},
        # brand-new match: full insert
        {"match_id": "20260612-brazil-haiti", "date": "2026-06-12",
         "home_team": "Brazil", "away_team": "Haiti", "home_score": 3,
         "away_score": 0, "tournament": "FIFA World Cup", "neutral": True,
         "city": "Miami", "country": "United States", "stage": None,
         "round": None, "group_label": None, "source": "upstream"},
    ]
    db.upsert_matches(con, rows)

    # existing row: score filled, but labels + source PRESERVED
    got = con.execute(
        "SELECT home_score, away_score, stage, round, group_label, source "
        "FROM matches WHERE match_id='20260611-france-senegal'"
    ).fetchone()
    assert got == (1, 0, "group", "MD1", "A", "wc2026")

    # new row: inserted with its own source
    new = con.execute(
        "SELECT home_score, away_score, source FROM matches "
        "WHERE match_id='20260612-brazil-haiti'"
    ).fetchone()
    assert new == (3, 0, "upstream")


def test_upsert_matches_empty_is_noop(con):
    db.upsert_matches(con, [])
    assert con.execute("SELECT count(*) FROM matches").fetchone()[0] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_db.py -k upsert_matches -q`
Expected: FAIL — `module 'wc2026.db' has no attribute 'upsert_matches'`.

- [ ] **Step 3: Write minimal implementation**

```python
# wc2026/db.py  (append)
_MATCH_UPSERT_COLS = ["match_id", "date", "home_team", "away_team",
                      "home_score", "away_score", "tournament", "neutral",
                      "city", "country", "stage", "round", "group_label",
                      "source"]


def upsert_matches(con: duckdb.DuckDBPyConnection, rows) -> None:
    """Insert new matches (full metadata) and, on match_id conflict, update ONLY
    the scores. Existing labels/source/venue are preserved (martj42 backfill
    rows carry no WC labels). `rows` is a list of dicts or a DataFrame with the
    match columns; missing columns are filled with NULL."""
    df = pd.DataFrame(rows)
    if df.empty:
        return
    for c in _MATCH_UPSERT_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[_MATCH_UPSERT_COLS]
    con.register("_um", df)
    con.execute(
        f"INSERT INTO matches ({', '.join(_MATCH_UPSERT_COLS)}) "
        f"SELECT {', '.join(_MATCH_UPSERT_COLS)} FROM _um "
        "ON CONFLICT (match_id) DO UPDATE SET "
        "home_score = excluded.home_score, away_score = excluded.away_score"
    )
    con.unregister("_um")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_db.py -k upsert_matches -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add wc2026/db.py tests/test_db.py
git commit -m "Add db.upsert_matches (insert + score-only conflict update)"
```

---

### Task 5: `fetch.reconcile` + `ChangeReport`

**Files:**
- Create: `wc2026/fetch.py`
- Test: `tests/test_fetch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetch.py  (append)
# NOTE: `providers` and `MatchRecord` are already imported at the top of this
# file (Task 2). Import only the new modules here to avoid a redefinition lint.
from wc2026 import db, fetch


@pytest.fixture
def con():
    c = db.connect(":memory:")
    db.init_schema(c)
    yield c
    c.close()


def _rec(date, h, a, status, hs=None, as_=None, tour="FIFA World Cup"):
    return MatchRecord(date=date, home_team=h, away_team=a, status=status,
                       source="upstream", home_score=hs, away_score=as_,
                       tournament=tour, neutral=True)


def test_reconcile_inserts_updates_skips_and_is_idempotent(con):
    # existing unplayed match in the DB
    con.execute(
        "INSERT INTO matches (match_id, date, home_team, away_team, source) "
        "VALUES ('20260611-france-senegal', DATE '2026-06-11', 'France', "
        "'Senegal', 'wc2026')"
    )
    records = [
        _rec("2026-06-11", "France", "Senegal", "FINISHED", 1, 0),   # fills score
        _rec("2026-06-12", "Brazil", "Haiti", "FINISHED", 3, 0),     # new played
        _rec("2026-06-13", "Spain", "Japan", "SCHEDULED"),           # new fixture
        _rec("2026-06-14", "Italy", "Peru", "IN_PLAY"),              # new, not final -> still a fixture insert? no: only FINISHED writes scores
    ]
    rep = fetch.reconcile(con, records)
    assert rep.score_changes == [("20260611-france-senegal", (None, None), (1, 0))]
    assert "20260612-brazil-haiti" in rep.inserted
    assert "20260613-spain-japan" in rep.inserted          # unplayed fixture inserted
    # France score now in DB
    assert con.execute(
        "SELECT home_score FROM matches WHERE match_id='20260611-france-senegal'"
    ).fetchone()[0] == 1

    # second run with the same records: no changes
    rep2 = fetch.reconcile(con, records)
    assert rep2.score_changes == []
    assert rep2.inserted == []


def test_reconcile_skips_nonfinal_for_existing(con):
    con.execute(
        "INSERT INTO matches (match_id, date, home_team, away_team, source) "
        "VALUES ('20260611-france-senegal', DATE '2026-06-11', 'France', "
        "'Senegal', 'wc2026')"
    )
    rep = fetch.reconcile(con, [_rec("2026-06-11", "France", "Senegal", "IN_PLAY")])
    assert rep.skipped_nonfinal == ["20260611-france-senegal"]
    assert con.execute(
        "SELECT home_score FROM matches WHERE match_id='20260611-france-senegal'"
    ).fetchone()[0] is None


def test_reconcile_dry_run_writes_nothing(con):
    rep = fetch.reconcile(con, [_rec("2026-06-12", "Brazil", "Haiti", "FINISHED", 3, 0)],
                          write=False)
    assert "20260612-brazil-haiti" in rep.inserted
    assert con.execute("SELECT count(*) FROM matches").fetchone()[0] == 0
```

Decision encoded by the test: a NEW match that is not `FINISHED` (e.g. `SCHEDULED`) is inserted as an unplayed fixture (NULL score). A NEW match that IS `FINISHED` is inserted with its score. An EXISTING match is only touched when `FINISHED` and the score differs; any non-final status for an existing match is skipped. (`IN_PLAY` for a brand-new match inserts a NULL-score fixture, same as `SCHEDULED`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_fetch.py -k reconcile -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.fetch'`.

- [ ] **Step 3: Write minimal implementation**

```python
# wc2026/fetch.py
"""Provider-agnostic reconciliation: apply the write policy and report changes.

No network, no raw SQL: providers fetch, `db` writes, this decides what to write."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import db
from .providers import MatchRecord


@dataclass
class ChangeReport:
    inserted: list[str] = field(default_factory=list)             # match_ids
    score_changes: list[tuple] = field(default_factory=list)      # (mid, old, new)
    skipped_nonfinal: list[str] = field(default_factory=list)     # match_ids

    def summary(self) -> str:
        return (f"{len(self.inserted)} inserted, "
                f"{len(self.score_changes)} score change(s), "
                f"{len(self.skipped_nonfinal)} skipped (not final)")


def _row(mid: str, r: MatchRecord, *, played: bool) -> dict:
    return {
        "match_id": mid, "date": r.date,
        "home_team": r.home_team, "away_team": r.away_team,
        "home_score": r.home_score if played else None,
        "away_score": r.away_score if played else None,
        "tournament": r.tournament, "neutral": r.neutral,
        "city": r.city, "country": r.country,
        "stage": r.stage, "round": r.round, "group_label": r.group_label,
        "source": r.source,
    }


def reconcile(con, records, write: bool = True) -> ChangeReport:
    """Apply the write policy to `records` and (unless write=False) persist via
    db.upsert_matches. Final-only; overwrite a present score only when it
    differs; insert new matches/fixtures; never silently clobber."""
    current = {
        mid: (h, a) for mid, h, a in con.execute(
            "SELECT match_id, home_score, away_score FROM matches"
        ).fetchall()
    }
    rep = ChangeReport()
    to_write: list[dict] = []
    for r in records:
        mid = db.make_match_id(r.date, r.home_team, r.away_team)
        finished = r.status == "FINISHED"
        if mid not in current:
            to_write.append(_row(mid, r, played=finished))
            rep.inserted.append(mid)
            continue
        if not finished:
            rep.skipped_nonfinal.append(mid)
            continue
        old = current[mid]
        new = (r.home_score, r.away_score)
        if old != new:
            to_write.append(_row(mid, r, played=True))
            rep.score_changes.append((mid, old, new))
    if write and to_write:
        db.upsert_matches(con, to_write)
    return rep
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_fetch.py -q`
Expected: all passed (providers + reconcile).

- [ ] **Step 5: Commit**

```bash
git add wc2026/fetch.py tests/test_fetch.py
git commit -m "Add fetch.reconcile and ChangeReport (write policy)"
```

---

### Task 6: `scripts/fetch_results.py` CLI

**Files:**
- Create: `scripts/fetch_results.py`
- Test: `tests/test_fetch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetch.py  (append)
from scripts import fetch_results


def test_select_records_default_is_wc_2026_only():
    recs = [
        MatchRecord("2026-06-11", "France", "Senegal", "FINISHED", "upstream",
                    tournament="FIFA World Cup"),
        MatchRecord("2024-01-01", "Spain", "Italy", "FINISHED", "upstream",
                    tournament="Friendly"),
        MatchRecord("2022-12-18", "Argentina", "France", "FINISHED", "upstream",
                    tournament="FIFA World Cup"),  # 2022, not this WC
    ]
    default = fetch_results.select_records(recs, corpus_refresh=False)
    assert [r.home_team for r in default] == ["France"]
    full = fetch_results.select_records(recs, corpus_refresh=True)
    assert len(full) == 3


def test_build_provider_unknown_raises():
    with pytest.raises(SystemExit):
        fetch_results.build_provider("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_fetch.py -k "select_records or build_provider" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.fetch_results'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/fetch_results.py
"""Fetch match results from an online source into the DuckDB store.

    uv run python -m scripts.fetch_results                 # WC-2026 via martj42
    uv run python -m scripts.fetch_results --corpus-refresh
    uv run python -m scripts.fetch_results --source live   # (not yet implemented)
    uv run python -m scripts.fetch_results --dry-run

Designed to run before scripts.run_schedule in the /reforecast workflow.
"""
from __future__ import annotations

import argparse

from wc2026 import db, fetch
from wc2026.providers import LiveApiProvider, Martj42CsvProvider, MatchRecord


def build_provider(source: str):
    if source == "martj42":
        return Martj42CsvProvider()
    if source == "live":
        return LiveApiProvider()
    raise SystemExit(f"unknown source: {source!r} (expected 'martj42' or 'live')")


def select_records(records: list[MatchRecord], corpus_refresh: bool
                   ) -> list[MatchRecord]:
    """Default: only this World Cup's matches (fast, focused). With
    corpus_refresh: the entire feed (catch historical additions/corrections)."""
    if corpus_refresh:
        return list(records)
    return [r for r in records
            if r.tournament == "FIFA World Cup" and str(r.date).startswith("2026")]


def main(source: str = "martj42", corpus_refresh: bool = False,
         dry_run: bool = False) -> None:
    provider = build_provider(source)
    records = select_records(provider.fetch(), corpus_refresh)
    con = db.connect(db.DB_PATH)
    try:
        report = fetch.reconcile(con, records, write=not dry_run)
    finally:
        con.close()
    print(report.summary() + (" (dry-run, nothing written)" if dry_run else ""))
    for mid, old, new in report.score_changes:
        print(f"  score: {mid} {old} -> {new}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["martj42", "live"], default="martj42")
    ap.add_argument("--corpus-refresh", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(args.source, args.corpus_refresh, args.dry_run)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_fetch.py -q`
Expected: all passed.

- [ ] **Step 5: End-to-end dry-run against the real feed**

Run: `uv run python -m scripts.fetch_results --dry-run`
Expected: a network call to martj42, then a summary line like
`N inserted, M score change(s), 0 skipped (not final) (dry-run, nothing written)`.
The DB is unchanged (dry-run). If offline, skip this step and note it.

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch_results.py tests/test_fetch.py
git commit -m "Add fetch_results CLI"
```

---

### Task 7: Documentation

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `.claude/commands/reforecast.md`

- [ ] **Step 1: Update `CLAUDE.md`**

Under the "Data store" section, add the fetcher facts (no code):
- `wc2026/providers.py` is the only network-touching module: `ResultsProvider`
  protocol + `MatchRecord`; `Martj42CsvProvider` (concrete) and `LiveApiProvider`
  (stub, reads `WC_RESULTS_API_KEY`, concrete API TBD).
- `wc2026/fetch.py` `reconcile()` holds the write policy (final-only,
  overwrite-if-changed, report); `db` stays policy-free. `db.upsert_matches`
  inserts new matches and updates only scores on conflict.
- `scripts/fetch_results.py`: manual command; default pulls this WC's results
  via martj42, `--corpus-refresh` does the full feed, `--dry-run` reports only.

- [ ] **Step 2: Update `README.md`**

In the "Data store" subsection, add the fetch workflow:
```bash
uv run python -m scripts.fetch_results            # pull WC results (martj42)
uv run python -m scripts.fetch_results --dry-run  # preview, write nothing
```
Note the live-API path is pluggable and not yet implemented.

- [ ] **Step 3: Update `.claude/commands/reforecast.md`**

Add a step 0 before the existing steps: optionally run
`uv run python -m scripts.fetch_results` to pull the latest results into the DB
before refitting (manual; surfaces what changed).

- [ ] **Step 4: Run the full suite**

Run: `uv run python -m pytest -q`
Expected: all passed (existing 42 + new fetch/db tests).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md .claude/commands/reforecast.md
git commit -m "Document the auto-fetcher"
```

---

## Notes for the implementer

- **No live network in tests.** `Martj42CsvProvider` takes an injectable
  `http_get` (default real httpx); tests pass a lambda returning sample CSV.
  Only Task 6 Step 5 hits the real feed, and it is a dry-run.
- **`reconcile` reads all current scores once** into a dict (fast even for the
  ~49k-row corpus) then writes in one `upsert_matches` call.
- **DuckDB `ON CONFLICT`** requires the conflict target to be the PK
  (`match_id`); `excluded.<col>` refers to the proposed row. Score-only update
  is deliberate (see the plan's refinement note).
- **`source` lives on `MatchRecord`** (each provider stamps it); inserts use it,
  conflict-updates never touch it.
- The live provider stays a stub this cycle; the interface, write path, and CLI
  are all exercised so it is a drop-in later.
```
