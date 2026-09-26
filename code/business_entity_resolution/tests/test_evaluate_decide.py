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


def test_threshold_cut():
    assert threshold_cut(np.array([0.2, 0.6]), 0.5).tolist() == [False, True]
