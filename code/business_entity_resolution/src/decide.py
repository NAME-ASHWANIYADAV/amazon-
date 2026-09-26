"""Turn pair probabilities into per-S1 match lists."""
import numba
import numpy as np

TAIL_CAP = 30  # support of the Poisson approximation for hidden positives beyond max_k


def assign_best(sx, p):
    """Every SX belongs to at most one S1: keep only its highest-probability pair."""
    order = np.lexsort((-p, sx))
    sx_sorted = sx[order]
    first = np.r_[True, sx_sorted[1:] != sx_sorted[:-1]]
    keep = np.zeros(len(p), dtype=bool)
    keep[order[first]] = True
    return keep


def threshold_cut(p, t):
    return p >= t


@numba.njit(cache=True)
def _ef_values(q, max_k, tail_cap):
    """Exact expected F0.5 of predicting the top-k (k = 0..m) of one S1's candidates.

    q: probabilities sorted descending. Candidates beyond m = min(len(q), max_k) cannot be predicted
    but may still be true: their count is modelled as Poisson(sum of their p) and enters recall."""
    n = len(q)
    m = min(n, max_k)
    lam = 0.0
    for j in range(m, n):
        lam += q[j]
    R = m + tail_cap + 1
    suf = np.zeros((m + 1, R))  # suf[k, r] = P(r true among candidates k..end)
    pk = np.exp(-lam)
    for r in range(tail_cap + 1):
        suf[m, r] = pk
        pk = pk * lam / (r + 1)
    for k in range(m - 1, -1, -1):
        for r in range(R):
            v = suf[k + 1, r] * (1.0 - q[k])
            if r > 0:
                v += suf[k + 1, r - 1] * q[k]
            suf[k, r] = v
    ef = np.zeros(m + 1)
    ef[0] = suf[0, 0]
    pre = np.zeros(m + 1)  # pre[t] = P(t true among the first k)
    pre[0] = 1.0
    for k in range(1, m + 1):
        qk = q[k - 1]
        for t in range(k, 0, -1):
            pre[t] = pre[t] * (1.0 - qk) + pre[t - 1] * qk
        pre[0] = pre[0] * (1.0 - qk)
        val = 0.0
        for t in range(1, k + 1):
            if pre[t] == 0.0:
                continue
            acc = 0.0
            for r in range(R):
                w = suf[k, r]
                if w > 0.0:
                    acc += w / (0.25 * (t + r) + k)
            val += pre[t] * 1.25 * t * acc
        ef[k] = val
    return ef


@numba.njit(parallel=True, cache=True)
def _best_k(p_sorted, starts, counts, max_k, tail_cap):
    best = np.zeros(len(starts), dtype=np.int64)
    for g in numba.prange(len(starts)):
        ef = _ef_values(p_sorted[starts[g]:starts[g] + counts[g]], max_k, tail_cap)
        best[g] = np.argmax(ef)
    return best


def expected_f_values(q_sorted, max_k=12):
    """E[F0.5] for k = 0..min(len, max_k) predicted matches of one S1 (q sorted descending)."""
    return _ef_values(np.asarray(q_sorted, dtype=np.float64), max_k, TAIL_CAP)


def expected_f05_cut(s1, p, max_k=12):
    """Per S1, predict the top-k pairs (by p) where k maximises exact expected F0.5 under independent
    Bernoulli(p) labels; k = 0 is chosen when 'no match' is the better bet."""
    order = np.lexsort((-p, s1))
    s_sorted, p_sorted = s1[order], np.asarray(p, dtype=np.float64)[order]
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    counts = np.diff(np.r_[starts, len(s_sorted)])
    best_k = _best_k(p_sorted, starts, counts, max_k, TAIL_CAP)
    rank = np.arange(len(s_sorted)) - np.repeat(starts, counts)
    mask = np.zeros(len(p), dtype=bool)
    mask[order] = rank < np.repeat(best_k, counts)
    return mask
