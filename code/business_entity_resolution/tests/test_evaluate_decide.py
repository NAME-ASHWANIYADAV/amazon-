import numpy as np
import pytest

from src.decide import assign_best, expected_f05_cut, threshold_cut
from src.evaluate import f05, macro_f05


def test_f05_organizer_example():
    assert f05({"a", "b", "c"}, {"a", "c"}) == pytest.approx(0.7142857, abs=1e-6)


def test_f05_singleton_rules():
    assert f05(set(), set()) == 1.0
    assert f05({"x"}, set()) == 0.0
    assert f05(set(), {"x"}) == 0.0


def test_macro_f05_averages_all_s1():
    assert macro_f05({"s1": {"a"}}, {"s1": {"a"}, "s2": set()}, ["s1", "s2"]) == 1.0


def test_assign_best_keeps_top_s1_per_sx():
    sx = np.array([1, 1, 2])
    p = np.array([0.3, 0.9, 0.5])
    assert assign_best(sx, p).tolist() == [False, True, True]


def test_expected_f05_cut_picks_confident_and_skips_singleton():
    s1 = np.array([0, 0, 0, 1])
    p = np.array([0.95, 0.9, 0.05, 0.02])
    assert expected_f05_cut(s1, p).tolist() == [True, True, False, False]


def test_expected_f05_cut_counts_tail_mass_beyond_max_k():
    # 1 strong-ish candidate + 39 weak ones: exact E[F] favours k=1 (0.333) over k=0 (0.183) only
    # when the weak tail's chance of hidden true matches is counted.
    s1 = np.zeros(40, dtype=np.int64)
    p = np.array([0.40] + [0.03] * 39)
    mask = expected_f05_cut(s1, p)
    assert mask.sum() == 1 and mask[0]


def test_expected_f05_cut_is_deterministic_and_order_independent():
    rng = np.random.default_rng(1)
    s1 = rng.integers(0, 50, 400)
    p = rng.random(400)
    a = expected_f05_cut(s1, p)
    perm = rng.permutation(400)
    b = expected_f05_cut(s1[perm], p[perm])
    assert (a[perm] == b).all()


def test_expected_f05_exact_value_small_case():
    from src.decide import expected_f_values
    # two candidates p=[0.9, 0.5]: E[F1] = 0.9*0.5*1.25/(0.5+1) + 0.9*0.5*1.25/(0.25+1)
    ef = expected_f_values(np.array([0.9, 0.5]))
    assert ef[0] == pytest.approx(0.1 * 0.5, abs=1e-9)
    assert ef[1] == pytest.approx(0.9 * 0.5 * 1.25 / 1.5 + 0.9 * 0.5 * 1.25 / 1.25, abs=1e-9)
    assert ef[2] == pytest.approx(0.45 * 1.0 + 0.9 * 0.5 * 1.25 * 1 / 2.25 + 0.1 * 0.5 * 1.25 / 2.25, abs=1e-9)


def test_threshold_cut():
    assert threshold_cut(np.array([0.2, 0.6]), 0.5).tolist() == [False, True]
