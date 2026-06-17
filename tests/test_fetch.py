import pytest

from scripts import fetch_results
from wc2026 import db, fetch, providers
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


def test_martj42_provider_handles_blank_neutral_and_missing_optionals():
    csv = (
        "date,home_team,away_team,home_score,away_score,tournament,neutral\n"
        "2026-06-11,France,Senegal,1,0,FIFA World Cup,\n"  # blank neutral, no city/country cols
    )
    p = providers.Martj42CsvProvider(http_get=lambda url: csv)
    (rec,) = p.fetch()
    assert rec.neutral is False        # blank -> not neutral, no bool(<NA>) crash
    assert rec.city is None            # column absent -> None
    assert rec.country is None


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
        _rec("2026-06-14", "Italy", "Peru", "IN_PLAY"),              # new, non-final
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


def test_reconcile_dedupes_duplicate_match_id_in_batch(con):
    # same match listed twice in one batch -> counted once, last value wins
    records = [
        _rec("2026-06-12", "Brazil", "Haiti", "FINISHED", 1, 0),
        _rec("2026-06-12", "Brazil", "Haiti", "FINISHED", 3, 0),
    ]
    rep = fetch.reconcile(con, records)
    assert rep.inserted == ["20260612-brazil-haiti"]   # once, not twice
    assert con.execute(
        "SELECT home_score, away_score FROM matches WHERE match_id='20260612-brazil-haiti'"
    ).fetchone() == (3, 0)                              # last wins


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
