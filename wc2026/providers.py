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


def _opt(row, name: str):
    """An optional CSV column: missing column or NA cell -> None."""
    value = getattr(row, name, None)
    return None if value is None or pd.isna(value) else value


def _httpx_get(url: str) -> str:
    import httpx

    resp = httpx.get(url, timeout=30.0)
    resp.raise_for_status()
    return resp.text


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
        # A blank/NA neutral cell is treated as not-neutral (fillna), so a sparse
        # feed never trips bool(<NA>).
        df["neutral"] = (
            df["neutral"].astype("string").str.upper().eq("TRUE").fillna(False)
        )
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
                tournament=_opt(r, "tournament"),
                neutral=bool(r.neutral),
                city=_opt(r, "city"),
                country=_opt(r, "country"),
            ))
        return out


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
