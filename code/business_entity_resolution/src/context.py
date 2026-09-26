"""Generator-aware context features found by error analysis on validation.

- House numbers: lookalike distractors shift the S1 number UP by one of DISTRACTOR_SHIFTS; true copies
  get symmetric +/-1..2 typos, truncations or zero padding. A shift followed by a truncation is a
  distractor signature. When a true copy's number differs from S1, other true copies usually share it.
- Empty-address copies: whether an empty-address record belongs to this S1 depends on how many
  same-name empty-address records compete in the list and how common the name is globally.
- Extra / missing name words: descriptor words (group, holdings, overseas, ...) mark lookalikes; generic
  suffixes (center, services, ...) are true-copy noise. Scored by out-of-fold log-odds from training pairs.
All inputs are per-record token arrays (features.build_token_arrays) and candidate-table columns.
"""
import numba
import numpy as np

DISTRACTOR_SHIFTS = np.array([1, 2, 3, 4, 5, 7, 9, 11, 13, 21], dtype=np.int64)
# French name words mapped to the English words whose extra/missing-word log-odds were learned on US/India
# training pairs (the country name inserted into a name plays the role of 'india'). Used only when carrying
# word log-odds to a test vocabulary; hand-written language knowledge.
WORD_EQUIV = {"france": "india", "groupe": "group", "holding": "holdings", "fils": "sons", "freres": "brothers",
              "cie": "co", "compagnie": "company", "associes": "associates", "centre": "center",
              "developpement": "development", "etablissements": "enterprises", "ets": "enterprises",
              "internationale": "international", "societe": "company", "services": "services",
              "technologies": "technologies", "solutions": "solutions", "gestion": "management",
              "conseil": "consulting", "partenaires": "partners", "industries": "industries"}
NO_NUM = 999.0

CTX_COLS = ["hn_off", "hn_in_set", "hn_neg", "hn_trunc", "hn_composite", "hn_sib_x", "hn_sib_1",
            "emp_dup_cnt", "same_core_in_list", "far_twin_cnt", "g_n_s1_core", "g_n_sx_core",
            "g_n_sxemp_core", "g_emp_ratio", "xw_lo_max", "xw_lo_sum", "xw_n_extra", "mw_lo_max", "mw_n_miss"]


@numba.njit(cache=True)
def core_hash(ptr, ids):
    """Order-free hash of each record's (sorted, unique) name token ids; 0 for an empty name."""
    n = len(ptr) - 1
    out = np.zeros(n, dtype=np.int64)
    for r in range(n):
        h = np.uint64(1469598103934665603)
        for k in range(ptr[r], ptr[r + 1]):
            h = (h ^ np.uint64(ids[k] + 1)) * np.uint64(1099511628211)
        out[r] = np.int64(h & np.uint64(0x7FFFFFFFFFFFFFFF)) if ptr[r + 1] > ptr[r] else 0
    return out


@numba.njit(cache=True)
def _ndigits(v):
    d = 1
    while v >= 10:
        v //= 10
        d += 1
    return d


@numba.njit(cache=True)
def _is_trunc(a, b):
    """b is a with its last digit dropped or its leading digit dropped."""
    if a < 10:
        return False
    if b == a // 10:
        return True
    return b == a % (10 ** (_ndigits(a) - 1))


@numba.njit(parallel=True, cache=True)
def number_alignment(a_ptr, a_val, b_ptr, b_val, pa, pb, shifts, off, bval, flags):
    """Best-aligned (S1 number a, SX number b) per pair (min |b - a|).
    off: signed b - a clipped to +/-60 (NO_NUM when a side has no numbers); bval: aligned b (-1 if none);
    flags[:, 0] in distractor set, 1 negative, 2 truncation of a, 3 composite shift+truncation."""
    for q in numba.prange(len(pa)):
        i, j = pa[q], pb[q]
        x0, x1, y0, y1 = a_ptr[i], a_ptr[i + 1], b_ptr[j], b_ptr[j + 1]
        if x1 == x0 or y1 == y0:
            off[q] = NO_NUM
            bval[q] = -1
            continue
        best, ba, bb = 1 << 62, 0, 0
        for x in range(x0, x1):
            for y in range(y0, y1):
                d = abs(b_val[y] - a_val[x])
                if d < best:
                    best, ba, bb = d, a_val[x], b_val[y]
        o = bb - ba
        off[q] = max(-60.0, min(60.0, float(o)))
        bval[q] = bb
        in_set = 0.0
        for s in shifts:
            if o == s:
                in_set = 1.0
        flags[q, 0] = in_set
        flags[q, 1] = 1.0 if o < 0 else 0.0
        flags[q, 2] = 1.0 if (o != 0 and _is_trunc(ba, bb)) else 0.0
        comp = 0.0
        if o != 0:
            for s in shifts:
                if _is_trunc(ba + s, bb):
                    comp = 1.0
        flags[q, 3] = comp


@numba.njit(cache=True)
def list_context(s1r, off, bval, cos_name, cos_addr, sx_empty, sx_key, s1_key, out):
    """Within each S1 list (rows grouped by s1): out columns
    0 hn_sib_x: other name-similar rows with the same aligned SX number (-1 if no number)
    1 hn_sib_1: other name-similar rows whose number equals the S1 number
    2 emp_dup_cnt: empty-address rows with this row's core name (incl. itself)
    3 same_core_in_list: rows with this row's core name
    4 far_twin_cnt: non-empty rows named like the S1 whose address differs (cos_addr < 0.5)."""
    n = len(s1r)
    i = 0
    while i < n:
        j = i
        while j < n and s1r[j] == s1r[i]:
            j += 1
        k1 = s1_key[i]
        far = 0
        n_eq1 = 0
        for u in range(i, j):
            if sx_key[u] == k1 and sx_key[u] != 0 and not sx_empty[u] and cos_addr[u] < 0.5:
                far += 1
            if cos_name[u] >= 0.75 and off[u] == 0.0:
                n_eq1 += 1
        for t in range(i, j):
            sib = 0
            emp = 0
            same = 0
            for u in range(i, j):
                if sx_key[u] == sx_key[t] and sx_key[t] != 0:
                    same += 1
                    if sx_empty[u]:
                        emp += 1
                if u != t and bval[t] >= 0 and bval[u] == bval[t] and cos_name[u] >= 0.75:
                    sib += 1
            out[t, 0] = sib if bval[t] >= 0 else -1.0
            out[t, 1] = n_eq1 - (1 if (cos_name[t] >= 0.75 and off[t] == 0.0) else 0)
            out[t, 2] = emp
            out[t, 3] = same
            out[t, 4] = far
        i = j


@numba.njit(parallel=True, cache=True)
def _extra_missing(a_ptr, a_ids, b_ptr, b_ids, pa, pb, lo_extra, lo_miss, out):
    """Sorted token-id sets A (S1) and B (SX): extra = B - A, missing = A - B.
    out: max lo over extra, sum lo over extra, n_extra, max lo over missing, n_missing."""
    for q in numba.prange(len(pa)):
        i, j = pa[q], pb[q]
        x, xe, y, ye = a_ptr[i], a_ptr[i + 1], b_ptr[j], b_ptr[j + 1]
        emax, esum, ne, mmax, nm = -9.0, 0.0, 0, -9.0, 0
        while x < xe or y < ye:
            if y >= ye or (x < xe and a_ids[x] < b_ids[y]):
                v = lo_miss[a_ids[x]]
                mmax = max(mmax, v)
                nm += 1
                x += 1
            elif x >= xe or b_ids[y] < a_ids[x]:
                v = lo_extra[b_ids[y]]
                emax = max(emax, v)
                esum += v
                ne += 1
                y += 1
            else:
                x += 1
                y += 1
        out[q, 0] = emax if ne > 0 else 0.0
        out[q, 1] = esum
        out[q, 2] = ne
        out[q, 3] = mmax if nm > 0 else 0.0
        out[q, 4] = nm


@numba.njit(cache=True)
def _extra_missing_counts(a_ptr, a_ids, b_ptr, b_ids, pa, pb, label, n_vocab, cnt):
    """cnt[token, 0/1/2/3] = extra-in-false, extra-in-true, missing-in-false, missing-in-true."""
    for q in range(len(pa)):
        i, j = pa[q], pb[q]
        x, xe, y, ye = a_ptr[i], a_ptr[i + 1], b_ptr[j], b_ptr[j + 1]
        lab = label[q]
        while x < xe or y < ye:
            if y >= ye or (x < xe and a_ids[x] < b_ids[y]):
                cnt[a_ids[x], 2 + lab] += 1
                x += 1
            elif x >= xe or b_ids[y] < a_ids[x]:
                cnt[b_ids[y], lab] += 1
                y += 1
            else:
                x += 1
                y += 1


def word_log_odds(tok, s1r, sxr, label, min_count=5):
    """Per-token log-odds (false vs true) of appearing as an extra SX word / a missing S1 word."""
    n_vocab = len(tok["idf_n"])
    cnt = np.zeros((n_vocab, 4), dtype=np.int64)
    _extra_missing_counts(tok["ns1_ptr"], tok["ns1_ids"], tok["nsx_ptr"], tok["nsx_ids"],
                          s1r.astype(np.int64), sxr.astype(np.int64), label.astype(np.int64), n_vocab, cnt)
    tot = cnt.sum(0).astype(np.float64)

    def lo(f, t):
        v = np.log((cnt[:, f] + 1) / (tot[f] + 2)) - np.log((cnt[:, t] + 1) / (tot[t] + 2))
        return np.where(cnt[:, f] + cnt[:, t] >= min_count, v, 0.0).astype(np.float32)

    return lo(0, 1), lo(2, 3)


def extra_word_features(tok, s1r, sxr, lo_extra, lo_miss):
    out = np.zeros((len(s1r), 5), dtype=np.float32)
    _extra_missing(tok["ns1_ptr"], tok["ns1_ids"], tok["nsx_ptr"], tok["nsx_ids"], s1r.astype(np.int64),
                   sxr.astype(np.int64), lo_extra, lo_miss, out)
    return out


_MASK60 = (1 << 60) - 1


def _with_country(h, cc):
    """Core-name hash made country-specific (bits 60-61 = country code); 0 (empty name) stays 0."""
    return np.where(h == 0, 0, (h & _MASK60) | (cc.astype(np.int64) << 60))


def global_name_counts(tok, s1_present=None):
    """Per-record counts over the whole split: S1s with the same core name, SX with it, empty-address SX.
    s1_present (bool per S1 row, optional): only these S1s count (the simulated-drop training view).
    With tok["cc_s1"/"cc_sx"] (country codes) names are counted within their own country."""
    h1 = core_hash(tok["ns1_ptr"], tok["ns1_ids"])
    hx = core_hash(tok["nsx_ptr"], tok["nsx_ids"])
    if "cc_s1" in tok:
        h1, hx = _with_country(h1, tok["cc_s1"]), _with_country(hx, tok["cc_sx"])
    if s1_present is not None:
        h1 = np.where(s1_present, h1, -1)   # -1 never equals a core hash (hashes are >= 0)
    empty = tok["flags_sx"][:, 2] > 0

    def counts(keys, values):
        u, c = np.unique(values, return_counts=True)
        if len(u) == 0:
            return np.zeros(len(keys), dtype=np.float32)
        pos = np.searchsorted(u, keys)
        pos = np.clip(pos, 0, len(u) - 1)
        return np.where(u[pos] == keys, c[pos], 0).astype(np.float32)

    return h1, hx, counts, empty


S2_COLS = ["p1_logit", "s2_n_conf", "s2_n_conf_nonempty", "s2_rank_p", "s2_max_other_p", "s2_nla",
           "s2_same_core_conf", "s2_sib_conf", "s2_sum_other_p"]


@numba.njit(cache=True)
def _stage2(s1r, p1, sx_empty, sx_key, s1_key, bval, out):
    """Within each S1 list (rows grouped by s1), context from stage-1 probabilities (see S2_COLS[1:])."""
    n = len(s1r)
    i = 0
    while i < n:
        j = i
        while j < n and s1r[j] == s1r[i]:
            j += 1
        nla = 0
        for u in range(i, j):
            if not sx_empty[u] and sx_key[u] == s1_key[u] and sx_key[u] != 0 and p1[u] < 0.5:
                nla += 1
        for t in range(i, j):
            n_conf, n_conf_ne, rank, same, sib = 0, 0, 0, 0, 0
            mx, sm = 0.0, 0.0
            for u in range(i, j):
                if p1[u] > p1[t]:
                    rank += 1
                if u == t:
                    continue
                sm += p1[u]
                if p1[u] > mx:
                    mx = p1[u]
                if p1[u] >= 0.5:
                    n_conf += 1
                    if not sx_empty[u]:
                        n_conf_ne += 1
                    if sx_key[u] == sx_key[t] and sx_key[t] != 0:
                        same += 1
                    if bval[t] >= 0 and bval[u] == bval[t]:
                        sib += 1
            out[t, 0] = n_conf
            out[t, 1] = n_conf_ne
            out[t, 2] = rank
            out[t, 3] = mx
            out[t, 4] = nla
            out[t, 5] = same
            out[t, 6] = sib
            out[t, 7] = sm
        i = j


def stage2_features(tok, s1r, sxr, p1):
    """S2_COLS for pairs grouped by s1, from stage-1 (out-of-fold) probabilities p1."""
    s1r64, sxr64 = s1r.astype(np.int64), sxr.astype(np.int64)
    n = len(s1r64)
    off = np.empty(n, dtype=np.float32)
    bval = np.empty(n, dtype=np.int64)
    flags = np.zeros((n, 4), dtype=np.float32)
    number_alignment(tok["nums1_ptr"], tok["nums1_val"], tok["numsx_ptr"], tok["numsx_val"], s1r64, sxr64,
                     DISTRACTOR_SHIFTS, off, bval, flags)
    h1 = core_hash(tok["ns1_ptr"], tok["ns1_ids"])
    hx = core_hash(tok["nsx_ptr"], tok["nsx_ids"])
    empty = tok["flags_sx"][:, 2] > 0
    out = np.zeros((n, len(S2_COLS)), dtype=np.float32)
    p = np.clip(p1.astype(np.float64), 1e-6, 1 - 1e-6)
    out[:, 0] = np.log(p / (1 - p))
    tmp = np.zeros((n, len(S2_COLS) - 1), dtype=np.float32)
    _stage2(s1r64, p1.astype(np.float32), empty[sxr64], hx[sxr64], h1[s1r64], bval, tmp)
    out[:, 1:] = tmp
    return out


def context_features(tok, s1r, sxr, cos_name, cos_addr, lo_extra, lo_miss, s1_present=None):
    """All CTX_COLS for candidate pairs grouped by s1 (s1r sorted/grouped). Returns float32 (n, len(CTX_COLS)).
    s1_present: see global_name_counts."""
    s1r64, sxr64 = s1r.astype(np.int64), sxr.astype(np.int64)
    n = len(s1r64)
    out = np.zeros((n, len(CTX_COLS)), dtype=np.float32)
    col = {c: i for i, c in enumerate(CTX_COLS)}
    off = np.empty(n, dtype=np.float32)
    bval = np.empty(n, dtype=np.int64)
    flags = np.zeros((n, 4), dtype=np.float32)
    number_alignment(tok["nums1_ptr"], tok["nums1_val"], tok["numsx_ptr"], tok["numsx_val"], s1r64, sxr64,
                     DISTRACTOR_SHIFTS, off, bval, flags)
    out[:, col["hn_off"]] = off
    out[:, col["hn_in_set"]:col["hn_composite"] + 1] = flags
    h1, hx, counts, empty = global_name_counts(tok, s1_present)
    sx_key, s1_key = hx[sxr64], h1[s1r64]
    lc = np.zeros((n, 5), dtype=np.float32)
    list_context(s1r64, off, bval, cos_name.astype(np.float32), cos_addr.astype(np.float32), empty[sxr64],
                 sx_key, s1_key, lc)
    out[:, col["hn_sib_x"]:col["far_twin_cnt"] + 1] = lc
    n_s1 = counts(sx_key, h1)
    n_sx = counts(sx_key, hx)
    n_emp = counts(sx_key, hx[empty])
    out[:, col["g_emp_ratio"]] = n_emp / np.maximum(n_s1, 1)
    if "cc_s1" in tok:
        # counts as rates of the country's size (S1s per 100k present S1s, SX per 1M SX): raw counts scale with
        # the split (test US has 0.62x the S1s of train US), which made test names look rare to the judge
        pres = np.ones(len(h1), dtype=bool) if s1_present is None else s1_present
        n1c = np.bincount(tok["cc_s1"][pres], minlength=4).astype(np.float64)
        nxc = np.bincount(tok["cc_sx"], minlength=4).astype(np.float64)
        c = tok["cc_s1"][s1r64]
        n_s1 = n_s1 * (1e5 / np.maximum(n1c, 1))[c]
        n_sx = n_sx * (1e6 / np.maximum(nxc, 1))[c]
        n_emp = n_emp * (1e6 / np.maximum(nxc, 1))[c]
    out[:, col["g_n_s1_core"]] = n_s1
    out[:, col["g_n_sx_core"]] = n_sx
    out[:, col["g_n_sxemp_core"]] = n_emp
    out[:, col["xw_lo_max"]:col["mw_n_miss"] + 1] = extra_word_features(tok, s1r64, sxr64, lo_extra, lo_miss)
    return out
