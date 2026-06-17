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
