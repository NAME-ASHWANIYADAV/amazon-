import numpy as np
import pytest

from src.context import (CTX_COLS, DISTRACTOR_SHIFTS, NO_NUM, context_features, core_hash, list_context,
                         number_alignment, word_log_odds)


def _csr(rows, dtype=np.int64):
    ptr = np.zeros(len(rows) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(r) for r in rows])
    return ptr, np.array([v for r in rows for v in r], dtype=dtype)


def test_number_alignment_offsets_and_flags():
    a_ptr, a_val = _csr([[4230], [216], [365], [100]])
    b_ptr, b_val = _csr([[4243], [214], [65], []])
    pa = np.array([0, 1, 2, 3])
    pb = np.array([0, 1, 2, 3])
    off = np.empty(4, np.float32)
    bval = np.empty(4, np.int64)
    flags = np.zeros((4, 4), np.float32)
    number_alignment(a_ptr, a_val, b_ptr, b_val, pa, pb, DISTRACTOR_SHIFTS, off, bval, flags)
    assert off.tolist() == [13.0, -2.0, -60.0, NO_NUM]
    assert flags[0].tolist() == [1, 0, 0, 0]          # +13: distractor shift
    assert flags[1].tolist() == [0, 1, 0, 0]          # -2: negative (true-copy noise)
    assert flags[2, 2] == 1.0                         # 365 -> 65: head truncation
    assert bval.tolist()[:3] == [4243, 214, 65] and bval[3] == -1


def test_composite_shift_then_truncation():
    a_ptr, a_val = _csr([[1417]])
    b_ptr, b_val = _csr([[142]])                      # 1417 + 3 = 1420, tail-truncated to 142
    off = np.empty(1, np.float32)
    bval = np.empty(1, np.int64)
    flags = np.zeros((1, 4), np.float32)
    number_alignment(a_ptr, a_val, b_ptr, b_val, np.array([0]), np.array([0]), DISTRACTOR_SHIFTS, off, bval, flags)
    assert flags[0, 3] == 1.0


def test_list_context_siblings_and_empty_duplicates():
    s1r = np.array([0, 0, 0, 0])
    off = np.array([2, 2, 0, NO_NUM], np.float32)
    bval = np.array([18, 18, 16, -1])
    cos_name = np.array([0.9, 0.9, 0.9, 0.95], np.float32)
    cos_addr = np.array([0.9, 0.9, 0.9, -1], np.float32)
    empty = np.array([False, False, False, True])
    sx_key = np.array([7, 7, 7, 7])
    s1_key = np.array([7, 7, 7, 7])
    out = np.zeros((4, 5), np.float32)
    list_context(s1r, off, bval, cos_name, cos_addr, empty, sx_key, s1_key, out)
    assert out[:, 0].tolist() == [1, 1, 0, -1]        # two records agree on 18
    assert out[:, 1].tolist() == [1, 1, 0, 1]         # one record matches the S1 number exactly
    assert out[:, 2].tolist() == [1, 1, 1, 1]         # one empty-address copy of the name
    assert out[:, 3].tolist() == [4, 4, 4, 4]


def test_word_log_odds_and_context_features_shapes():
    # vocab: 0 hari, 1 exports, 2 group, 3 center
    ns1_ptr, ns1_ids = _csr([[0, 1]], np.int32)
    nsx_ptr, nsx_ids = _csr([[0, 1, 2], [0, 1, 3], [0, 1]], np.int32)
    tok = {"ns1_ptr": ns1_ptr, "ns1_ids": ns1_ids, "nsx_ptr": nsx_ptr, "nsx_ids": nsx_ids,
           "idf_n": np.ones(4, np.float32), "nums1_ptr": _csr([[5]])[0], "nums1_val": _csr([[5]])[1],
           "numsx_ptr": _csr([[5], [4], []])[0], "numsx_val": _csr([[5], [4], []])[1],
           "flags_sx": np.zeros((3, 4), np.float32)}
    s1r = np.zeros(30, dtype=np.int64)
    sxr = np.array([0, 1, 2] * 10)
    label = np.array([0, 1, 1] * 10)
    lo_extra, lo_miss = word_log_odds(tok, s1r, sxr, label)
    assert lo_extra[2] > 1.0 and lo_extra[3] < 0          # 'group' extra only in false, 'center' only in true
    f = context_features(tok, np.array([0, 0, 0]), np.array([0, 1, 2]), np.ones(3, np.float32),
                         np.ones(3, np.float32), lo_extra, lo_miss)
    col = {c: i for i, c in enumerate(CTX_COLS)}
    assert f.shape == (3, len(CTX_COLS))
    assert f[0, col["xw_lo_max"]] == pytest.approx(lo_extra[2])
    assert f[2, col["xw_n_extra"]] == 0 and f[1, col["hn_neg"]] == 1.0


def test_global_name_counts_lookup():
    from src.context import global_name_counts
    ns1_ptr, ns1_ids = _csr([[0], [0], [1]], np.int32)
    nsx_ptr, nsx_ids = _csr([[0], [2], [0], [1]], np.int32)
    tok = {"ns1_ptr": ns1_ptr, "ns1_ids": ns1_ids, "nsx_ptr": nsx_ptr, "nsx_ids": nsx_ids,
           "flags_sx": np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], np.float32)}
    h1, hx, counts, empty = global_name_counts(tok)
    assert counts(hx, h1).tolist() == [2, 0, 2, 1]        # S1s sharing each SX core name
    assert counts(hx, hx[empty]).tolist() == [2, 0, 2, 0]  # empty-address SX sharing it


def test_core_hash_order_free_and_empty():
    ptr, ids = _csr([[1, 2], [1, 2], [2, 3], []], np.int32)
    h = core_hash(ptr, ids)
    assert h[0] == h[1] != h[2] and h[3] == 0
