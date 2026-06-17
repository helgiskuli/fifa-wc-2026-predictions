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
from wc2026.providers import (LiveApiProvider, Martj42CsvProvider, MatchRecord,
                              ResultsProvider)


def build_provider(source: str) -> ResultsProvider:
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
    # A dry-run only reads current scores to compute the diff: open read-only so
    # it writes nothing and can coexist with other readers.
    con = db.connect(db.DB_PATH, read_only=dry_run)
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
