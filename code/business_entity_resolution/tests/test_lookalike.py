import numpy as np

from src.lookalike import extra_legal, lookalike_reject


def _cols(off, ins, sib1, sibx, s3, xw=None):
    n = len(off)
    return {"hn_off": np.array(off, np.float32), "hn_in_set": np.array(ins, np.float32),
            "hn_sib_1": np.array(sib1, np.float32), "hn_sib_x": np.array(sibx, np.float32),
            "cos_name": np.full(n, 0.9, np.float32), "xw_n_extra": np.array(xw or [0] * n, np.float32),
            "mw_n_miss": np.zeros(n, np.float32), "is_s3": np.array(s3, np.float32)}


def test_r1_same_source_conflict_takes_the_whole_group():
    # one S1: anchors at the S1 number from S2 and S3, a +1 group (S2 + S3), a +7 single, a word fake
    s1r = np.zeros(6, np.int64)
    p = np.array([.99, .98, .9, .85, .8, .9], np.float32)
    keep = np.ones(6, bool)
    cols = _cols(off=[0, 0, 1, 1, 7, 3], ins=[0, 0, 1, 1, 1, 1], sib1=[1, 1, 2, 2, 2, 0], sibx=[0, 0, 1, 1, 0, 0],
                 s3=[0, 1, 0, 1, 0, 0])
    xword = np.array([0, 0, 0, 0, 0, 1], bool)
    rej, m = lookalike_reject(s1r, p, keep, cols, np.zeros(6, bool), xword)
    assert m["R1"].tolist() == [False, False, True, True, False, False]
    assert m["RM"].tolist() == [False, False, True, True, False, False]   # the +1 group mixes S2 and S3
    assert m["W"].tolist() == [False] * 5 + [True]
    assert rej.tolist() == [False, False, True, True, False, True]


def test_ra_needs_every_group_member_modified():
    s1r = np.zeros(3, np.int64)
    p = np.array([.99, .9, .9], np.float32)
    keep = np.ones(3, bool)
    cols = _cols(off=[0, 2, 2], ins=[0, 1, 1], sib1=[0, 1, 1], sibx=[0, 1, 1], s3=[1, 0, 0])
    xleg = np.array([False, True, False])     # one member of the +2 group is an unmodified name -> RA stays off
    _, m = lookalike_reject(s1r, p, keep, cols, xleg, np.zeros(3, bool), rules=("RA",))
    assert not m["RA"].any()
    xleg = np.array([False, True, True])
    _, m = lookalike_reject(s1r, p, keep, cols, xleg, np.zeros(3, bool), rules=("RA",))
    assert m["RA"].tolist() == [False, True, True]


def test_extra_legal():
    got = extra_legal(["", "llc"], ["inc", "llc", ""], np.array([0, 1, 1]), np.array([0, 1, 0]))
    assert got.tolist() == [True, False, True]


def test_word_rule_only_at_a_shifted_number_and_rf_only_for_unseen_countries():
    s1r = np.zeros(4, np.int64)
    p = np.array([.99, .9, .9, .9], np.float32)
    keep = np.ones(4, bool)
    cols = _cols(off=[0, 0, 1, 2], ins=[0, 0, 1, 1], sib1=[1, 1, 1, 1], sibx=[0, 0, 0, 0], s3=[0, 1, 1, 1],
                 xw=[0, 1, 1, 0])
    xword = np.array([False, True, False, False])            # 'france' added at the exact address: kept
    xleg = np.array([False, False, False, True])
    for unseen, want in ((False, [False] * 4), (True, [False, False, True, True])):
        _, m = lookalike_reject(s1r, p, keep, cols, xleg, xword, ldrop=np.zeros(4, bool),
                                unseen=np.full(4, unseen))
        assert not m["W"].any()
        assert m["RF"].tolist() == want


def test_legal_swapped_and_rl_only_for_strict_countries():
    from src.lookalike import legal_swapped
    got = legal_swapped(["llc", "", "inc"], ["ltd", "llc", "inc"], np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert got.tolist() == [True, False, False]          # swap, addition, same
    s1r = np.zeros(2, np.int64)
    p = np.array([.99, .9], np.float32)
    cols = _cols(off=[0, 1], ins=[0, 1], sib1=[1, 1], sibx=[0, 0], s3=[0, 1])
    xleg = np.array([False, True])
    for strict, want in ((False, [False, False]), (True, [False, True])):
        _, m = lookalike_reject(s1r, p, np.ones(2, bool), cols, xleg, np.zeros(2, bool), strict=np.full(2, strict))
        assert m["RL"].tolist() == want
