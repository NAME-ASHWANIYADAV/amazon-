"""Turn pair probabilities into per-S1 match lists."""
import numpy as np


def assign_best(sx, p):
    """Every SX belongs to at most one S1: keep only its highest-probability pair."""
    order = np.lexsort((-p, sx))
    first = np.r_[True, sx[order][1:] != sx[order][:-1]]
    keep = np.zeros(len(p), dtype=bool)
    keep[order[first]] = True
    return keep


def threshold_cut(p, t):
    return p >= t


def expected_f05_cut(s1, p, max_k=12, n_samples=256, seed=0, chunk=20_000):
    """Per S1, predict the top-k pairs (by p) where k maximises expected F0.5 under independent
    Bernoulli(p) labels; k = 0 is chosen when 'no match' is the better bet."""
    order = np.lexsort((-p, s1))
    s_sorted, p_sorted = s1[order], p[order]
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    counts = np.diff(np.r_[starts, len(s_sorted)])
    rank = np.arange(len(s_sorted)) - np.repeat(starts, counts)
    G = len(starts)
    P = np.zeros((G, max_k), dtype=np.float32)
    m = rank < max_k
    P[np.repeat(np.arange(G), counts)[m], rank[m]] = p_sorted[m]
    rng = np.random.default_rng(seed)
    best_k = np.zeros(G, dtype=np.int64)
    ks = np.arange(1, max_k + 1, dtype=np.float32)
    for g in range(0, G, chunk):
        Pc = P[g:g + chunk]
        Z = rng.random((len(Pc), n_samples, max_k), dtype=np.float32) < Pc[:, None, :]
        T = Z.sum(-1, dtype=np.float32)
        TP = np.cumsum(Z, -1, dtype=np.float32)
        F = (1.25 * TP / (0.25 * T[..., None] + ks)).mean(1)
        F0 = (T == 0).mean(1, dtype=np.float32)
        best_k[g:g + chunk] = np.argmax(np.concatenate([F0[:, None], F], 1), 1)
    pick_sorted = rank < np.repeat(best_k, counts)
    mask = np.zeros(len(p), dtype=bool)
    mask[order] = pick_sorted
    return mask
