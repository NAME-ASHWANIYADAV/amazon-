"""Pair features for the judge: embedding cosines, token-set overlaps, fuzzy scores, numbers, flags."""
from array import array

import numba
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

FEATURES = [
    "cos", "cos_name", "cos_addr", "rank", "gap_best", "z_in_list", "list_size",
    "name_inter", "name_len1", "name_lenx", "name_jacc", "name_cont1", "name_contx", "name_idf_jacc",
    "name_idf_inter", "tset", "tsort", "partial", "ratio", "nospace", "jw", "alt_best", "legal_code",
    "addr_inter", "addr_len1", "addr_lenx", "addr_jacc", "addr_idf_jacc", "addr_tset", "addr_ratio",
    "num_primary_eq", "num_any_eq", "num_log_mindiff", "num_min_rel", "num_n1", "num_nx",
    "num_x_unmatched", "num_x_primary_in_1", "num_prim_logdiff", "num_prim_rel",
    "was_indic", "is_domain", "addr_empty_x", "is_s3", "acro"]
NUM_COLS = ["num_primary_eq", "num_any_eq", "num_log_mindiff", "num_min_rel", "num_n1", "num_nx",
            "num_x_unmatched", "num_x_primary_in_1", "num_prim_logdiff", "num_prim_rel"]
# Cross-S1 competition: how this S1 compares with every other S1 whose candidate list contains the same SX.
COMP_COLS = ["comp_n_other", "comp_rank", "comp_margin", "comp_n_close", "comp_margin_name", "comp_is_best"]
# Features precomputed on the candidate table and copied into X: list context, competition, generator-aware
# context (context.CTX_COLS) and the stage-A filter probability.
from .context import CTX_COLS  # noqa: E402

LIST_COLS = ["cos", "cos_name", "cos_addr", "rank", "gap_best", "z_in_list", "list_size"]
STAGE_A_COLS = LIST_COLS + COMP_COLS + ["addr_empty_x"]
FEATURES = FEATURES + COMP_COLS + CTX_COLS + ["pA"]
TABLE_COLS = LIST_COLS + COMP_COLS + CTX_COLS + ["pA"]


def build_vocab(strings_iter):
    """Token -> id in exactly the order build_token_arrays assigns name ids (s1 then sx, token order)."""
    vocab = {}
    for strings in strings_iter:
        for s in strings:
            if s:
                for t in s.split():
                    vocab.setdefault(t, len(vocab))
    return vocab


@numba.njit(cache=True)
def _competition(sx, cos, name, orig, n_keep, out):
    """Rows sorted by (sx, cos desc); orig = original row index. Writes out[orig] for orig < n_keep.
    out columns: n_other, rank, margin, n_close, margin_name, is_best."""
    n = len(sx)
    i = 0
    while i < n:
        j = i
        while j < n and sx[j] == sx[i]:
            j += 1
        b1, b1i, b2 = -9.0, -1, -9.0  # top-2 name cosine in the group
        for t in range(i, j):
            if name[t] > b1:
                b2, b1, b1i = b1, name[t], t
            elif name[t] > b2:
                b2 = name[t]
        g = j - i
        e = i
        for t in range(i, j):
            other = (cos[i + 1] if g > 1 else -9.0) if t == i else cos[i]
            thr = cos[t] - 0.02
            while e < j and cos[e] >= thr:
                e += 1
            o = orig[t]
            if o >= n_keep:
                continue
            other_name = b2 if t == b1i else b1
            out[o, 0] = g - 1
            out[o, 1] = t - i
            out[o, 2] = cos[t] - other if other > -9.0 else 1.0
            out[o, 3] = e - i - 1
            out[o, 4] = name[t] - other_name if other_name > -9.0 else 1.0
            out[o, 5] = 1.0 if t == i else 0.0
        i = j


COMPETITOR_RANK = 20  # an S1 competes for an SX when the SX is in its top-20 record-cosine candidates


@numba.njit(parallel=True, cache=True)
def _competition_vs(q_s1, q_sx, q_cos, q_name, t_sx, t_s1, t_cos, t_name, t_start, t_end, u_sx, out):
    """For each query row, compare with competitor rows of the same SX from OTHER S1s.
    Competitor table is sorted by sx; u_sx are its unique sx values with [t_start, t_end) ranges."""
    for q in numba.prange(len(q_s1)):
        lo, hi = 0, len(u_sx)
        x = q_sx[q]
        while lo < hi:  # binary search for the SX group
            mid = (lo + hi) // 2
            if u_sx[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        n_other, rank, close = 0, 0, 0
        best, best_name = -9.0, -9.0
        if lo < len(u_sx) and u_sx[lo] == x:
            for k in range(t_start[lo], t_end[lo]):
                if t_s1[k] == q_s1[q]:
                    continue
                n_other += 1
                if t_cos[k] > best:
                    best = t_cos[k]
                if t_name[k] > best_name:
                    best_name = t_name[k]
                if t_cos[k] > q_cos[q]:
                    rank += 1
                if t_cos[k] >= q_cos[q] - 0.02:
                    close += 1
        out[q, 0] = n_other
        out[q, 1] = rank
        out[q, 2] = q_cos[q] - best if n_other > 0 else 1.0
        out[q, 3] = close
        out[q, 4] = q_name[q] - best_name if n_other > 0 else 1.0
        out[q, 5] = 1.0 if rank == 0 else 0.0


def competition_vs_table(q_s1, q_sx, q_cos, q_name, t_s1, t_sx, t_cos, t_name):
    """COMP_COLS for query pairs against a competitor table (each S1's top-COMPETITOR_RANK candidates):
    other S1s claiming the same SX, this pair's rank among them, margins and near-ties."""
    order = np.argsort(t_sx, kind="stable")
    t_sx, t_s1 = t_sx[order].astype(np.int64), t_s1[order].astype(np.int64)
    t_cos, t_name = t_cos[order].astype(np.float32), t_name[order].astype(np.float32)
    u_sx, t_start, counts = np.unique(t_sx, return_index=True, return_counts=True)
    out = np.zeros((len(q_s1), len(COMP_COLS)), dtype=np.float32)
    _competition_vs(q_s1.astype(np.int64), q_sx.astype(np.int64), q_cos.astype(np.float32),
                    q_name.astype(np.float32), t_sx, t_s1, t_cos, t_name, t_start.astype(np.int64),
                    (t_start + counts).astype(np.int64), u_sx, out)
    return out


def competition_features(sx, cos, cos_name, n_keep=None):
    """COMP_COLS for rows [0, n_keep) (original order). Must be given ALL candidate rows of the split,
    i.e. every S1's list, so that each SX sees all of its competing S1s; rows >= n_keep only act as
    competitors (keeps the output small when extra competitor lists are appended)."""
    n_keep = len(sx) if n_keep is None else n_keep
    order = np.lexsort((-cos, sx))
    out = np.zeros((n_keep, len(COMP_COLS)), dtype=np.float32)
    _competition(sx[order], cos[order].astype(np.float32), cos_name[order].astype(np.float32), order,
                 n_keep, out)
    return out


def _token_csr(strings, vocab, skip_digits=False):
    """Space-separated tokens -> (indptr, sorted unique int32 ids); vocab grows in place."""
    indptr = np.zeros(len(strings) + 1, dtype=np.int64)
    ids = array("i")
    for r, s in enumerate(strings):
        if s:
            ids.extend(sorted({vocab.setdefault(t, len(vocab)) for t in s.split()
                               if not (skip_digits and t.isdigit())}))
        indptr[r + 1] = len(ids)
    return indptr, np.frombuffer(ids, dtype=np.int32).copy()


def _number_csr(strings):
    indptr = np.zeros(len(strings) + 1, dtype=np.int64)
    vals = array("q")
    for r, s in enumerate(strings):
        if s:
            vals.extend(int(t[:15]) for t in s.split())
        indptr[r + 1] = len(vals)
    return indptr, np.frombuffer(vals, dtype=np.int64).copy()


def add_list_features(cand):
    """Per-S1 list context (needs the whole candidate list, so run before chunking).
    `cand` must be grouped by s1 (it is sorted by s1, rank)."""
    s1r, cos = cand["s1"].to_numpy(), cand["cos"].to_numpy().astype(np.float32)
    starts = np.flatnonzero(np.r_[True, s1r[1:] != s1r[:-1]])
    counts = np.diff(np.r_[starts, len(s1r)])
    mx = np.maximum.reduceat(cos, starts)
    mu = np.add.reduceat(cos, starts) / counts
    sd = np.sqrt(np.maximum(np.add.reduceat(cos * cos, starts) / counts - mu * mu, 0))
    rep = lambda v: np.repeat(v, counts).astype(np.float32)
    return cand.with_columns(pl.Series("gap_best", rep(mx) - cos),
                             pl.Series("z_in_list", (cos - rep(mu)) / (rep(sd) + 1e-3)),
                             pl.Series("list_size", rep(counts)))


COUNTRY_CODES = {"France": 0, "India": 1, "US": 2}   # 3 = anything else
N_CODES = 4


def country_idf(ptr, ids, rec_cc, vocab_size):
    """idf per country (rows = country code) over that country's SX records. One idf over the whole split is
    not comparable between train and test: test US has 0.62x the SX of train US while the split total barely
    changes, which inflated every US token's idf by ~0.45 and made US names look rarer (over-matching)."""
    rec_of_tok = np.repeat(np.arange(len(ptr) - 1, dtype=np.int32), np.diff(ptr))
    cc = rec_cc[rec_of_tok]
    out = np.empty((N_CODES, vocab_size), dtype=np.float32)
    for c in range(N_CODES):
        n_docs = int((rec_cc == c).sum())
        df = np.bincount(ids[cc == c], minlength=vocab_size).astype(np.float32)
        out[c] = np.log((n_docs + 1) / (df + 1)) + 1.0
    return out


@numba.njit(parallel=True, cache=True)
def _set_feats(a_ptr, a_ids, b_ptr, b_ids, idf, cc, pa, pb, out):
    """out: inter, |A|, |B|, idf(A∩B), idf(A∪B). idf: (countries, vocab); cc: country code per pair."""
    for q in numba.prange(len(pa)):
        i, j = pa[q], pb[q]
        c = cc[q]
        x, xe, y, ye = a_ptr[i], a_ptr[i + 1], b_ptr[j], b_ptr[j + 1]
        inter, wi, wu = 0, 0.0, 0.0
        while x < xe and y < ye:
            if a_ids[x] == b_ids[y]:
                inter += 1
                wi += idf[c, a_ids[x]]
                wu += idf[c, a_ids[x]]
                x += 1
                y += 1
            elif a_ids[x] < b_ids[y]:
                wu += idf[c, a_ids[x]]
                x += 1
            else:
                wu += idf[c, b_ids[y]]
                y += 1
        while x < xe:
            wu += idf[c, a_ids[x]]
            x += 1
        while y < ye:
            wu += idf[c, b_ids[y]]
            y += 1
        out[q, 0] = inter
        out[q, 1] = a_ptr[i + 1] - a_ptr[i]
        out[q, 2] = b_ptr[j + 1] - b_ptr[j]
        out[q, 3] = wi
        out[q, 4] = wu


@numba.njit(parallel=True, cache=True)
def _num_feats(a_ptr, a_val, b_ptr, b_val, pa, pb, out):
    """out: primary_eq, any_eq, log1p(min|diff|), min rel diff, n_a, n_b, b_unmatched, b_primary_in_a,
    log1p(|primary_a - primary_b|), relative primary diff. -1 marks 'not applicable' (no numbers)."""
    for q in numba.prange(len(pa)):
        i, j = pa[q], pb[q]
        x0, x1, y0, y1 = a_ptr[i], a_ptr[i + 1], b_ptr[j], b_ptr[j + 1]
        out[q, 4] = x1 - x0
        out[q, 5] = y1 - y0
        if x1 == x0 or y1 == y0:
            out[q, 0] = -1.0
            out[q, 1] = -1.0
            out[q, 2] = -1.0
            out[q, 3] = -1.0
            out[q, 7] = -1.0
            out[q, 8] = -1.0
            out[q, 9] = -1.0
            out[q, 6] = y1 - y0
            continue
        pd = float(abs(a_val[x0] - b_val[y0]))
        out[q, 8] = np.log1p(pd)
        out[q, 9] = pd / max(a_val[x0], b_val[y0], 1)
        prim = a_val[x0]
        peq, anyeq, unmatched = 0, 0, 0
        best, brel = 1e18, 1e18
        for y in range(y0, y1):
            v = b_val[y]
            if v == prim:
                peq = 1
            found = 0
            for x in range(x0, x1):
                u = a_val[x]
                if u == v:
                    found = 1
                d = float(abs(u - v))
                if d < best:
                    best = d
                r = d / max(u, v, 1)
                if r < brel:
                    brel = r
            if found:
                anyeq = 1
            else:
                unmatched += 1
        bprim = 0
        for x in range(x0, x1):
            if a_val[x] == b_val[y0]:
                bprim = 1
        out[q, 0] = peq
        out[q, 1] = anyeq
        out[q, 2] = np.log1p(best)
        out[q, 3] = brel
        out[q, 6] = unmatched
        out[q, 7] = bprim


def _legal_matrix(uniq):
    M = np.zeros((len(uniq), len(uniq)), dtype=np.float32)
    sets = [set(u.split()) for u in uniq]
    for a, sa in enumerate(sets):
        for b, sb in enumerate(sets):
            M[a, b] = (0 if not sa and not sb else 1 if sa == sb else 2 if not sa or not sb
                       else 3 if sa & sb else 4)
    return M


def _cp(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / 100.0


def build_token_arrays(load_col, log=print):
    """Per-record structures for one split, built one column at a time to bound peak memory.

    load_col(which, column) returns a list (string columns) or numpy array for which in {"s1", "sx"}.
    Returns a dict of numpy arrays (save with np.savez, reload with load_token_arrays)."""
    import gc
    tok = {}
    for field, col, skip in (("n", "name_core", False), ("a", "addr_norm", True)):
        vocab = {}
        for which in ("s1", "sx"):
            tok[f"{field}{which}_ptr"], tok[f"{field}{which}_ids"] = _token_csr(load_col(which, col), vocab,
                                                                                 skip_digits=skip)
            gc.collect()
        n_docs = len(tok[f"{field}sx_ptr"]) - 1
        df = np.bincount(tok[f"{field}sx_ids"], minlength=len(vocab)).astype(np.float32)
        tok[f"idf_{field}"] = (np.log((n_docs + 1) / (df + 1)) + 1.0).astype(np.float32)
        log(f"tokens {col}: vocab {len(vocab)}")
        del vocab
        gc.collect()
    for which in ("s1", "sx"):
        tok[f"num{which}_ptr"], tok[f"num{which}_val"] = _number_csr(load_col(which, "numbers"))
    l1, lx = load_col("s1", "legal"), load_col("sx", "legal")
    uniq = sorted(set(l1) | set(lx))
    code = {u: i for i, u in enumerate(uniq)}
    tok["legal_M"] = _legal_matrix(uniq)
    tok["legal_s1"] = np.array([code[u] for u in l1], dtype=np.int32)
    tok["legal_sx"] = np.array([code[u] for u in lx], dtype=np.int32)
    del l1, lx
    tok["flags_sx"] = np.stack([load_col("sx", c) for c in ("was_indic", "is_domain", "addr_empty")]
                               + [load_col("sx", "src") == 3], 1).astype(np.float32)
    return tok


def load_token_arrays(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def acronym_code(core_a, core_b):
    """1 if name core b (spaces removed, 2-6 letters) is the initials of core a's 2+ tokens, 2 if a is the
    initials of b, else 0. The generator abbreviates names this way ('tourcoing societe' -> 'ts'), far more
    often in France than in the training countries."""
    def is_acro(full, short):
        s = short.replace(" ", "")
        t = full.split()
        return len(t) >= 2 and 2 <= len(s) <= 6 and s.isalpha() and len(short.split()) <= 3 and \
            s == "".join(w[0] for w in t)
    return 1 if is_acro(core_a, core_b) else 2 if is_acro(core_b, core_a) else 0


class PairFeaturizer:
    """Computes features for candidate chunks from precomputed token arrays (build_token_arrays) plus
    the string columns rapidfuzz needs: s1[name_core, addr_norm], sx[name_core, alt_core, addr_norm]."""

    def __init__(self, s1, sx, tok):
        self.s1, self.sx = s1, sx
        self.n1, self.nx = (tok["ns1_ptr"], tok["ns1_ids"]), (tok["nsx_ptr"], tok["nsx_ids"])
        self.a1, self.ax = (tok["as1_ptr"], tok["as1_ids"]), (tok["asx_ptr"], tok["asx_ids"])
        self.num1, self.numx = (tok["nums1_ptr"], tok["nums1_val"]), (tok["numsx_ptr"], tok["numsx_val"])
        if "cc_s1" in tok:   # per-country idf (see country_idf)
            self.cc1 = tok["cc_s1"].astype(np.int64)
            self.idf_n = country_idf(tok["nsx_ptr"], tok["nsx_ids"], tok["cc_sx"], len(tok["idf_n"]))
            self.idf_a = country_idf(tok["asx_ptr"], tok["asx_ids"], tok["cc_sx"], len(tok["idf_a"]))
        else:
            self.cc1 = np.zeros(len(tok["ns1_ptr"]) - 1, dtype=np.int64)
            self.idf_n, self.idf_a = tok["idf_n"][None, :], tok["idf_a"][None, :]
        self.legal_M, self.legal1, self.legalx = tok["legal_M"], tok["legal_s1"], tok["legal_sx"]
        self.flags_x = tok["flags_sx"]

    def compute(self, cand):
        s1r, sxr = cand["s1"].to_numpy().astype(np.int64), cand["sx"].to_numpy().astype(np.int64)
        n = len(s1r)
        X = np.zeros((n, len(FEATURES)), dtype=np.float32)
        col = {f: i for i, f in enumerate(FEATURES)}
        # precomputed on the whole candidate table (list, competition, context, stage-A probability)
        for f in TABLE_COLS:
            X[:, col[f]] = cand[f].to_numpy()

        o = np.zeros((n, 5), dtype=np.float32)
        cc = self.cc1[s1r]
        _set_feats(self.n1[0], self.n1[1], self.nx[0], self.nx[1], self.idf_n, cc, s1r, sxr, o)
        X[:, col["name_inter"]], X[:, col["name_len1"]], X[:, col["name_lenx"]] = o[:, 0], o[:, 1], o[:, 2]
        X[:, col["name_jacc"]] = o[:, 0] / np.maximum(o[:, 1] + o[:, 2] - o[:, 0], 1)
        X[:, col["name_cont1"]] = o[:, 0] / np.maximum(o[:, 1], 1)
        X[:, col["name_contx"]] = o[:, 0] / np.maximum(o[:, 2], 1)
        X[:, col["name_idf_jacc"]] = o[:, 3] / np.maximum(o[:, 4], 1e-6)
        X[:, col["name_idf_inter"]] = o[:, 3]
        _set_feats(self.a1[0], self.a1[1], self.ax[0], self.ax[1], self.idf_a, cc, s1r, sxr, o)
        X[:, col["addr_inter"]], X[:, col["addr_len1"]], X[:, col["addr_lenx"]] = o[:, 0], o[:, 1], o[:, 2]
        X[:, col["addr_jacc"]] = o[:, 0] / np.maximum(o[:, 1] + o[:, 2] - o[:, 0], 1)
        X[:, col["addr_idf_jacc"]] = o[:, 3] / np.maximum(o[:, 4], 1e-6)

        on = np.zeros((n, len(NUM_COLS)), dtype=np.float32)
        _num_feats(self.num1[0], self.num1[1], self.numx[0], self.numx[1], s1r, sxr, on)
        for c, f in enumerate(NUM_COLS):
            X[:, col[f]] = on[:, c]

        c1 = self.s1["name_core"].gather(s1r).to_list()
        cx = self.sx["name_core"].gather(sxr).to_list()
        X[:, col["tset"]] = _cp(fuzz.token_set_ratio, c1, cx)
        X[:, col["tsort"]] = _cp(fuzz.token_sort_ratio, c1, cx)
        X[:, col["partial"]] = _cp(fuzz.partial_ratio, c1, cx)
        X[:, col["ratio"]] = _cp(fuzz.ratio, c1, cx)
        X[:, col["nospace"]] = _cp(fuzz.ratio, [s.replace(" ", "") for s in c1], [s.replace(" ", "") for s in cx])
        X[:, col["jw"]] = process.cpdist(c1, cx, scorer=JaroWinkler.normalized_similarity, workers=-1,
                                         dtype=np.float32)
        alt = X[:, col["tset"]].copy()
        altx = self.sx["alt_core"].gather(sxr).to_list()
        for q in np.flatnonzero(np.fromiter((bool(a) for a in altx), dtype=bool, count=n)):
            alt[q] = max(fuzz.token_set_ratio(c1[q], part) for part in altx[q].split("|")) / 100.0
        X[:, col["alt_best"]] = alt
        X[:, col["acro"]] = np.fromiter((acronym_code(a, b) for a, b in zip(c1, cx)), dtype=np.float32, count=n)
        X[:, col["legal_code"]] = self.legal_M[self.legal1[s1r], self.legalx[sxr]]
        a1 = self.s1["addr_norm"].gather(s1r).to_list()
        ax = self.sx["addr_norm"].gather(sxr).to_list()
        X[:, col["addr_tset"]] = _cp(fuzz.token_set_ratio, a1, ax)
        X[:, col["addr_ratio"]] = _cp(fuzz.ratio, a1, ax)
        X[:, col["was_indic"]:col["is_s3"] + 1] = self.flags_x[sxr]
        return X
