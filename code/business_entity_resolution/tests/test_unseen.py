import numpy as np

from src.unseen import _lo_from, train_counts_in_test_vocab


def test_lo_from_min_count_and_sign():
    cnt = np.array([[50, 5, 0, 0], [1, 1, 0, 0], [0, 0, 3, 30]], dtype=np.int64)
    tot = cnt.sum(0).astype(float)
    lo_extra, lo_miss = _lo_from(cnt, tot)
    assert lo_extra[0] > 0 and lo_extra[1] == 0        # fake-like word; too rare -> 0
    assert lo_miss[2] < 0 and lo_extra[2] == 0


def test_train_counts_map_exact_then_equivalent():
    # train vocab: group(0) holdings(1); test vocab: groupe(0) holdings(1) zzz(2)
    tok = {"ns1_ptr": np.array([0, 0]), "ns1_ids": np.array([], np.int64),
           "nsx_ptr": np.array([0, 1, 2]), "nsx_ids": np.array([0, 1])}
    j_s1 = np.array([0, 0]); j_sx = np.array([0, 1]); lab = np.array([0, 0])
    base, tot = train_counts_in_test_vocab(tok, j_s1, j_sx, lab, ["group", "holdings"], ["groupe", "holdings", "zzz"])
    assert base[0, 0] == 1       # 'groupe' -> WORD_EQUIV 'group' -> extra-in-false count
    assert base[1, 0] == 1       # exact token
    assert base[2].sum() == 0    # unknown word
    assert tot[0] == 2
