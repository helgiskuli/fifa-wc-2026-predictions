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
