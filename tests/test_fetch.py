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
