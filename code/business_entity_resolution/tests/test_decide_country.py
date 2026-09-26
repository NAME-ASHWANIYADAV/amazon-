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


def test_shift_rule_mask_reads_either_feature_layout():
    from src.features import FEATURES
    from src.run_pipeline import shift_rule_mask
    for names in (FEATURES, [f for f in FEATURES if f != "acro"]):
        X = np.zeros((4, len(names)), dtype=np.float16)
        i, o, s = names.index("hn_in_set"), names.index("hn_off"), names.index("hn_sib_1")
        X[:, i] = [1, 1, 1, 0]
        X[:, o] = [7, 2, 7, 7]      # shift 7, typo-size shift 2, shift 7, not in the distractor set
        X[:, s] = [1, 1, 0, 1]      # exact-number siblings confirm the S1 number (rows 0, 1, 3)
        assert shift_rule_mask(X).tolist() == [True, False, False, False]


def test_decide_reject_never_matches():
    s1r, sxr = np.array([0, 0]), np.array([10, 11])
    p = np.array([0.9, 0.95], dtype=np.float32)
    got = _decide(s1r, sxr, p, {"rule": "threshold", "threshold": 0.5}, np.array(["US"]),
                  reject=np.array([False, True]))
    assert got.tolist() == [True, False] and p.tolist()[1] == np.float32(0.95)


def test_cap_per_source_keeps_top_p_per_source():
    from src.run_pipeline import cap_per_source
    s1r = np.zeros(8, np.int64)
    is_s3 = np.array([0] * 7 + [1], bool)            # seven S2 matches, one S3
    p = np.array([.9, .8, .95, .7, .6, .85, .5, .4], np.float32)
    mask = np.ones(8, bool)
    got = cap_per_source(s1r, is_s3, p, mask, caps=(5, 6))
    assert got.tolist() == [True, True, True, True, False, True, False, True]   # the two lowest S2 go


def test_decide_post_rejects_after_assignment():
    s1r, sxr = np.array([0, 1]), np.array([10, 10])     # one SX claimed by two S1s
    p = np.array([0.9, 0.8], dtype=np.float32)
    got = _decide(s1r, sxr, p, {"rule": "threshold", "threshold": 0.5}, np.array(["US", "US"]),
                  post=lambda pp, keep: keep)          # reject whatever was assigned
    assert got.tolist() == [False, False]              # the SX does not flow to the second S1
