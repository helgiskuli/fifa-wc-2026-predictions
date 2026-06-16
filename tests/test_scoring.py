"""Office-pool point scoring (predict.score_prediction)."""
from wc2026 import ScoringConfig, score_prediction

CFG = ScoringConfig()  # a=3, b=1, c=1


def s(ph, pa, ah, aa):
    return score_prediction(ph, pa, ah, aa, CFG)


def test_exact_scores_six():
    # outcome(3) + both goal counts(1+1) + goal diff(1) = 6
    assert s(2, 1, 2, 1) == 6.0


def test_outcome_only():
    # home win predicted & actual, but wrong margin and goals
    assert s(1, 0, 3, 1) == CFG.a  # 3: outcome yes, diff no, neither count


def test_outcome_plus_one_goal_count():
    # 2-1 vs 2-0: home-win outcome(3) + home count(1); diff & away wrong
    assert s(2, 1, 2, 0) == CFG.a + CFG.b


def test_goal_difference_without_exact():
    # 2-1 vs 3-2: both home wins by 1 -> outcome(3)+diff(1); no count matches
    assert s(2, 1, 3, 2) == CFG.a + CFG.c


def test_wrong_outcome_scores_zero_unless_count_matches():
    # predict home win 1-0, actual away win 1-2: away count (0 vs 2) no,
    # home count 1==1 yes -> just the one goal count, no outcome/diff
    assert s(1, 0, 1, 2) == CFG.b


def test_draw_exact():
    assert s(0, 0, 0, 0) == 6.0


def test_draw_outcome_and_diff_but_not_exact():
    # 0-0 vs 1-1: draw outcome(3) + diff 0==0 (1); counts 0 vs 1 no
    assert s(0, 0, 1, 1) == CFG.a + CFG.c


def test_ratio_agnostic():
    cfg = ScoringConfig(a=5, b=2, c=1)
    assert score_prediction(2, 1, 2, 1, cfg) == 5 + 2 + 2 + 1  # exact = a+2b+c
