"""Provider-agnostic reconciliation: apply the write policy and report changes.

No network, no raw SQL: providers fetch, `db` writes, this decides what to write.
"""
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
