"""Transductive adaptation for countries absent from the training data (France in the test set).

The judge's word log-odds columns (xw_lo_*, mw_lo_max) are learned from the training countries' labels; in a
new country most descriptor words are unknown (lo = 0), and the judge accepts same-number one-word swaps that
it learned to trust for typo tokens. Measured on a labelled proxy (US-only judge scored on India) the penalty is
0.052 macro F0.5; 85% of the recoverable part comes back from two label-free steps:
1. pseudo word log-odds: pseudo-label the new country's pairs with the current decision, recount the word
   statistics (training truth + new-country pseudo labels, cross-fitted by S1 parity), rescore; 3 rounds
   (+0.023 on the proxy, a no-op on seen countries);
2. self-training: retrain the stage-1 judge on the training pairs plus the new country's confident pseudo
   labels, cross-fitted by S1 parity (+0.0065 on the proxy)."""
import numpy as np
import polars as pl
import xgboost as xgb

from . import judge
from .context import WORD_EQUIV, _extra_missing_counts, extra_word_features
from .decide import assign_best, expected_f05_cut
from .features import FEATURES

LO_COLS = [FEATURES.index(c) for c in ("xw_lo_max", "xw_lo_sum", "mw_lo_max")]
BAND = 0.05   # self-training uses pseudo labels only where p <= BAND or p >= 1 - BAND


def _counts(tok, s1, sx, lab, nv):
    cnt = np.zeros((nv, 4), dtype=np.int64)
    _extra_missing_counts(tok["ns1_ptr"], tok["ns1_ids"], tok["nsx_ptr"], tok["nsx_ids"], s1.astype(np.int64),
                          sx.astype(np.int64), lab.astype(np.int64), nv, cnt)
    return cnt


def _lo_from(cnt, tot, min_count=5):
    def lo(f, t):
        v = np.log((cnt[:, f] + 1) / (tot[f] + 2)) - np.log((cnt[:, t] + 1) / (tot[t] + 2))
        return np.where(cnt[:, f] + cnt[:, t] >= min_count, v, 0.0).astype(np.float32)
    return lo(0, 1), lo(2, 3)


def train_counts_in_test_vocab(tok_train, j_s1, j_sx, j_lab, vocab_train, vocab_test):
    """Training-truth word counts (extra/missing x false/true) re-indexed to the test vocabulary, exact token
    first, else its WORD_EQUIV image (the same mapping predict-test uses for the log-odds)."""
    cnt_tr = _counts(tok_train, j_s1, j_sx, j_lab, len(vocab_train))
    dtr = pl.DataFrame({"token": vocab_train, "id_tr": np.arange(len(vocab_train), dtype=np.int64)})
    dte = pl.DataFrame({"token": vocab_test, "id_te": np.arange(len(vocab_test), dtype=np.int64)})
    dte = dte.with_columns(pl.col("token").replace(WORD_EQUIV).alias("equiv"))
    m = (dte.join(dtr, on="token", how="left")
         .join(dtr.rename({"token": "equiv", "id_tr": "id_eq"}), on="equiv", how="left").sort("id_te"))
    map_ = m["id_tr"].fill_null(m["id_eq"]).fill_null(-1).to_numpy()
    base = np.zeros((len(vocab_test), 4), dtype=np.int64)
    ok = map_ >= 0
    base[ok] = cnt_tr[map_[ok]]
    return base, cnt_tr.sum(0).astype(np.float64)


def _decide(s1, sx, p):
    keep = assign_best(sx, p)
    m = np.zeros(len(p), dtype=bool)
    idx = np.flatnonzero(keep)
    m[idx[expected_f05_cut(s1[idx], p[idx])]] = True
    return m


def pseudo_lo_rounds(tok, s1, sx, X, models, cnt_base, tot_base, rounds=3, log=print):
    """s1/sx: the unseen country's pairs (grouped by s1); X: their feature rows (float32, modified in place:
    the lo columns receive the final pseudo log-odds); models: stage-1 fold judges. Returns stage-1 p."""
    nv = cnt_base.shape[0]
    fold = s1 % 2

    def lo_columns(lo_by_fold):
        L = np.zeros((len(s1), 3), dtype=np.float32)
        for f in (0, 1):
            mk = fold == f
            L[mk] = extra_word_features(tok, s1[mk], sx[mk], *lo_by_fold[f])[:, [0, 1, 3]]
        return L

    def predict():
        return np.mean([judge.predict(m, X) for m in models], axis=0)

    lo0 = _lo_from(cnt_base, tot_base)
    X[:, LO_COLS] = lo_columns({0: lo0, 1: lo0})
    p = predict()
    m0 = _decide(s1, sx, p)
    n_s1 = len(np.unique(s1))
    log(f"  pseudo-lo round 0: {m0.sum() / n_s1:.4f} matches/S1, lo-blind extra words {float((X[:, LO_COLS[0]] == 0).mean()):.3f}")
    for it in range(1, rounds + 1):
        lab = _decide(s1, sx, p)
        lo_f = {}
        for f in (0, 1):
            o = fold != f   # the other fold's pseudo labels count for this fold
            c = _counts(tok, s1[o], sx[o], lab[o], nv)
            lo_f[f] = _lo_from(cnt_base + c, tot_base + c.sum(0))
        X[:, LO_COLS] = lo_columns(lo_f)
        p = predict()
        mk = _decide(s1, sx, p)
        log(f"  pseudo-lo round {it}: {mk.sum() / n_s1:.4f} matches/S1, +{int((mk & ~m0).sum())} / -{int((~mk & m0).sum())}"
            f" vs round 0, lo-blind extra words {float((X[:, LO_COLS[0]] == 0).mean()):.3f}")
    return p


def self_train(XJ, yj, n_eval, s1, sx, X, p, log=print, band=BAND):
    """Stage-1 judges retrained on the training pairs plus the unseen country's confident pseudo labels (the
    current decision, where p <= band or p >= 1 - band) of one S1-parity fold, predicting the other fold.
    XJ/yj: training matrix and labels with the early-stopping slice as the LAST n_eval rows; X/p: the unseen
    country's rows (with pseudo log-odds) and stage-1 p."""
    lab = _decide(s1, sx, p).astype(np.float32)
    out = np.empty(len(p), dtype=np.float32)
    fold = s1 % 2
    conf = (p <= band) | (p >= 1 - band)
    for f in (0, 1):
        tr = conf & (fold == f)
        Xa = np.vstack([XJ[:-n_eval], X[tr], XJ[-n_eval:]])
        ya = np.r_[yj[:-n_eval], lab[tr], yj[-n_eval:]]
        m = judge.train(Xa, ya, n_eval)
        te = fold != f
        out[te] = judge.predict(m, X[te])
        log(f"  self-training fold {f}: {int(tr.sum())} pseudo rows added, best iteration {m.best_iteration}")
        del Xa, ya, m
    return out
