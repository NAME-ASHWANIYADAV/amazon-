import numpy as np

from src.run_pipeline import _country_map, _decide


def test_country_shift_moves_only_that_country_and_keeps_p():
    s1r = np.array([0, 0, 1, 1])
    sxr = np.array([10, 11, 12, 13])
    p = np.array([0.8, 0.75, 0.8, 0.75], dtype=np.float32)
    country = np.array(["France", "US"])
    base = {"rule": "threshold", "threshold": 0.7}
    assert _decide(s1r, sxr, p, base, country).tolist() == [True, True, True, True]
    shifted = dict(base, country_shift=_country_map("France=-0.9"))
    assert _decide(s1r, sxr, p, shifted, country).tolist() == [False, False, True, True]
    assert p.tolist() == [0.800000011920929, 0.75, 0.800000011920929, 0.75]   # caller's array untouched
    ef = {"rule": "expected_f", "threshold": 0.65, "country_shift": {"France": -3.0}}
    assert _decide(s1r, sxr, p, ef, country).tolist() == [False, False, True, True]


def test_country_map_parses_empty_and_lists():
    assert _country_map("") == {}
    assert _country_map("France=0.85,India=0.7") == {"France": 0.85, "India": 0.7}
