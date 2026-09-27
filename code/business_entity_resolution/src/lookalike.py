"""Reject lookalike distractor groups: fake businesses that copy an S1's name (plus a legal form or descriptor)
at a house number shifted by a distractor offset, and that come with 2+ copies in test (in train such fakes are
single records, so the judge reads a group of them as 'the S1 number is the noisy one' and trusts it).

Every rule uses a generator regularity measured on train truth:
- one entity keeps one house number per source, so a +1/+2 copy from the same source as a confident copy at
  the S1 number is not a copy (R1); the whole (S1, number) group it belongs to goes with it;
- true copies sharing a shifted number almost always come from one source, fakes mix S2 and S3 (RM);
- shifted groups whose every member only adds a legal form or descriptor are fakes (RA with a confirming copy
  at the S1 number, RB without one);
- descriptor words that are never a true copy in train (holdings, group, the country name) and their French
  analogues, at a shifted house number (W; at the exact address 'france' is true-copy noise like 'fils');
- in a country the model never saw in training, a +1/+2 copy whose name words or legal form changed while
  copies confirm the S1 number (RF). True number typos are sign-symmetric (+1/+2 as often as -1/-2 in V);
  test France has 11x more +1/+2 than -1/-2 predictions in exactly this cell.
Rejected pairs are removed after the SX -> best-S1 assignment (so the SX does not flow to another S1)."""
import numba
import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein

DISTRACTOR_SET = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
FAKE_WORDS = ("holding", "holdings", "group", "groupe", "participations")
STRONG_FAKE_WORDS = ("participations", "holding")   # ~0.2% at the exact address in unseen-country test data
COUNTRY_WORD = {"France": "france", "India": "india"}
NEED_COLS = ["hn_off", "hn_in_set", "hn_sib_1", "hn_sib_x", "cos_name", "xw_n_extra", "mw_n_miss", "is_s3",
             "addr_empty_x"]
ALL_RULES = ("R1", "RA", "RM", "RB", "W", "RF", "RL", "RW")
# words a true copy may swap in or add (copy noise in train and, by their exact-address profile, in France)
COPY_NOISE_WORDS = frozenset("center services service partners fils cie associes groupe developpement france".split())
COMMON_WORD_DF = 200   # a swapped-in word counts as 'common' when at least this many S1 names contain it
# default package: R1+RM+RB+W, plus RF (countries unseen in training) and RL (countries whose true copies almost
# never swap legal forms). Per-rule mirror accounting (+1/+2 vs -1/-2 predictions) finds each of them net positive;
# RA is left out (zero net in India, and it overlaps R1 elsewhere). Full-V cost of the package: -0.00018.
RULES = ("R1", "RM", "RB", "W", "RF", "RL")
LEGAL_SWAP_STRICT = 0.05   # RL applies where under 5% of train true copies swap the legal form (US 0.6%, India 21%)


def _token_flags(ptr, ids, token_ids):
    """bool (records, len(token_ids)): record's token list contains the token."""
    rec = np.repeat(np.arange(len(ptr) - 1, dtype=np.int64), np.diff(ptr))
    out = np.zeros((len(ptr) - 1, len(token_ids)), dtype=bool)
    for k, t in enumerate(token_ids):
        if t >= 0:
            out[rec[ids == t], k] = True
    return out


def extra_legal(s1_legal, sx_legal, s1r, sxr):
    """Pair flag: the SX carries a legal form that its S1 does not (prep 'legal' strings, space separated)."""
    uniq = sorted(set(s1_legal) | set(sx_legal))
    code = {u: i for i, u in enumerate(uniq)}
    sets = [set(u.split()) for u in uniq]
    M = np.array([[bool(b - a) for b in sets] for a in sets], dtype=bool)
    c1 = np.array([code[u] for u in s1_legal], dtype=np.int32)
    cx = np.array([code[u] for u in sx_legal], dtype=np.int32)
    return M[c1[s1r], cx[sxr]]


def legal_swapped(s1_legal, sx_legal, s1r, sxr):
    """Pair flag: both carry legal forms and the SX's differ from the S1's (a swap, e.g. llc -> ltd)."""
    uniq = sorted(set(s1_legal) | set(sx_legal))
    code = {u: i for i, u in enumerate(uniq)}
    sets = [set(u.split()) for u in uniq]
    M = np.array([[bool(a) and bool(b) and a != b for b in sets] for a in sets], dtype=bool)
    c1 = np.array([code[u] for u in s1_legal], dtype=np.int32)
    cx = np.array([code[u] for u in sx_legal], dtype=np.int32)
    return M[c1[s1r], cx[sxr]]


def extra_words(tok, vocab, s1r, sxr, pair_country):
    """Pair flag: the SX name adds a fake descriptor word, or its own country's name, that the S1 name lacks."""
    pos = {t: i for i, t in enumerate(vocab)}
    words = list(FAKE_WORDS) + sorted(set(COUNTRY_WORD.values()))
    ids = [pos.get(w, -1) for w in words]
    fx = _token_flags(tok["nsx_ptr"], tok["nsx_ids"], ids)
    f1 = _token_flags(tok["ns1_ptr"], tok["ns1_ids"], ids)
    added = fx[sxr] & ~f1[s1r]                                   # (pairs, words)
    out = added[:, :len(FAKE_WORDS)].any(axis=1)
    for c, w in COUNTRY_WORD.items():
        k = words.index(w)
        out |= added[:, k] & (pair_country == c)
    return out


@numba.njit(parallel=True, cache=True)
def _single_diff(a_ptr, a_ids, b_ptr, b_ids, pa, pb, out):
    """Sorted token-id sets A (S1) and B (SX): out[q] = (|B-A|, |A-B|, first id of B-A, first id of A-B)."""
    for q in numba.prange(len(pa)):
        i, j = pa[q], pb[q]
        x, xe, y, ye = a_ptr[i], a_ptr[i + 1], b_ptr[j], b_ptr[j + 1]
        ne, nm, fe, fm = 0, 0, -1, -1
        while x < xe or y < ye:
            if y >= ye or (x < xe and a_ids[x] < b_ids[y]):
                if nm == 0:
                    fm = a_ids[x]
                nm += 1
                x += 1
            elif x >= xe or b_ids[y] < a_ids[x]:
                if ne == 0:
                    fe = b_ids[y]
                ne += 1
                y += 1
            else:
                x += 1
                y += 1
        out[q, 0], out[q, 1], out[q, 2], out[q, 3] = ne, nm, fe, fm


def _subseq(a, b):
    if not a or not b or a[0] != b[0]:
        return False
    it = iter(b)
    return all(ch in it for ch in a)


def common_word_swap(tok, vocab, s1r, sxr, cand, df_min=COMMON_WORD_DF):
    """Pair flag (only where cand): the SX name is the S1 name with exactly one word swapped for a COMMON word
    (found in >= df_min S1 names) that is not a typo of the old word (Levenshtein similarity < 0.6), not an
    abbreviation of it, and not copy noise. In train (V) such swaps at the same house number are 90-99% fakes
    (lookalike businesses: 'amicale du team' -> 'comite du team'); test France predicts 30x more of them."""
    idx = np.flatnonzero(cand)
    d = np.zeros((len(idx), 4), dtype=np.int64)
    _single_diff(tok["ns1_ptr"], tok["ns1_ids"], tok["nsx_ptr"], tok["nsx_ids"], s1r[idx].astype(np.int64),
                 sxr[idx].astype(np.int64), d)
    one = (d[:, 0] == 1) & (d[:, 1] == 1)
    df1 = np.bincount(tok["ns1_ids"], minlength=len(vocab))
    out = np.zeros(len(s1r), dtype=bool)
    sub = np.flatnonzero(one & (df1[np.maximum(d[:, 2], 0)] >= df_min))
    for k in sub:
        e, m = vocab[d[k, 2]], vocab[d[k, 3]]
        if e in COPY_NOISE_WORDS or Levenshtein.normalized_similarity(e, m) >= 0.6 or _subseq(e, m) or _subseq(m, e):
            continue
        out[idx[k]] = True
    return out


def legal_dropped(s1_legal, sx_legal, s1r, sxr):
    """Pair flag: the S1 has a legal form and the SX has none (a symmetric, true-copy edit)."""
    e1 = np.array([not s for s in s1_legal], dtype=bool)
    ex = np.array([not s for s in sx_legal], dtype=bool)
    return ~e1[s1r] & ex[sxr]


BOOST_WORDS = ("groupe", "developpement", "france")
BOOST_P = 0.95
BOOST_COLS = ["xw_n_extra", "mw_n_miss", "addr_tset", "num_primary_eq", "addr_empty_x"]


def word_boost(tok, vocab, s1r, sxr, cols, unseen):
    """Pairs to raise to p >= BOOST_P: in a country unseen in training, the SX's ONLY extra name word is a
    dual-role word (fake descriptor at shifted numbers, true-copy noise at the exact address: 'groupe',
    'developpement', 'france'), at most one S1 word is missing, and the address is the same with the same house
    number. The word log-odds carried from English ('group' is always fake in train) push these true copies to
    p ~0.1; at the exact address they occur like the true-noise words fils/cie/services (~10k France pairs)."""
    pos = {t: i for i, t in enumerate(vocab)}
    ids = [pos.get(w, -1) for w in BOOST_WORDS]
    fx = _token_flags(tok["nsx_ptr"], tok["nsx_ids"], ids)
    f1 = _token_flags(tok["ns1_ptr"], tok["ns1_ids"], ids)
    added = (fx[sxr] & ~f1[s1r]).any(axis=1)
    c = {k: np.asarray(v, dtype=np.float32) for k, v in cols.items()}
    return (unseen & added & (c["xw_n_extra"] == 1) & (c["mw_n_miss"] <= 1) & (c["addr_tset"] >= 0.999)
            & (c["num_primary_eq"] > 0.5) & (c["addr_empty_x"] < 0.5))


def strong_fake_words(tok, vocab, s1r, sxr):
    """Pair flag: the SX name adds a word that marks a fake at any house number (STRONG_FAKE_WORDS)."""
    pos = {t: i for i, t in enumerate(vocab)}
    ids = [pos.get(w, -1) for w in STRONG_FAKE_WORDS]
    fx = _token_flags(tok["nsx_ptr"], tok["nsx_ids"], ids)
    f1 = _token_flags(tok["ns1_ptr"], tok["ns1_ids"], ids)
    return (fx[sxr] & ~f1[s1r]).any(axis=1)


def lookalike_reject(s1r, p, keep, cols, xleg, xword, rules=RULES, tau=0.5, ldrop=None, unseen=None, strict=None,
                     rw=None, xstrong=None):
    """cols: dict of NEED_COLS arrays per pair; xleg/xword: pair flags from extra_legal/extra_words;
    ldrop: pair flag from legal_dropped; unseen: pair flag 'S1 country not in the training data' (RF only);
    strict: pair flag 'S1 country's train true copies swap legal forms under LEGAL_SWAP_STRICT' (RL only).
    rw: pair flag from common_word_swap (RW only; the caller restricts it to unseen countries at offset 0).
    Returns (reject mask, {rule: mask}) over pairs; only kept (assigned) pairs are ever rejected."""
    n = len(p)
    off = np.asarray(cols["hn_off"], dtype=np.float32)
    ndiff = (np.asarray(cols["xw_n_extra"], dtype=np.float32) + np.asarray(cols["mw_n_miss"], dtype=np.float32)) > 0
    df = pl.DataFrame({
        "s1": np.asarray(s1r, dtype=np.int64), "p": np.asarray(p, dtype=np.float32), "keep": keep,
        "off": np.where(off >= 998, np.nan, off), "ins": np.asarray(cols["hn_in_set"]) > 0.5,
        "sib1": np.asarray(cols["hn_sib_1"], dtype=np.float32), "sibx": np.asarray(cols["hn_sib_x"], dtype=np.float32),
        "sim": np.asarray(cols["cos_name"], dtype=np.float32) >= 0.75, "s3": np.asarray(cols["is_s3"]) > 0.5,
        "mod": (np.asarray(cols["xw_n_extra"], dtype=np.float32) > 0) | xleg, "xword": xword,
        "xleg": xleg, "ndiff": ndiff,
        "ldrop": np.zeros(n, dtype=bool) if ldrop is None else ldrop,
        "unseen": np.zeros(n, dtype=bool) if unseen is None else unseen,
        "strict": np.zeros(n, dtype=bool) if strict is None else strict,
        "rw": np.zeros(n, dtype=bool) if rw is None else rw,
        "xstrong": np.zeros(n, dtype=bool) if xstrong is None else xstrong,
    }).with_columns(pl.col("off").fill_nan(None))
    num = pl.col("off").is_not_null() & (pl.col("off") != 0)
    anc = (pl.col("keep") & (pl.col("p") >= tau) & (pl.col("off") == 0)).fill_null(False).cast(pl.Int32)
    grp = pl.col("sim") & num
    df = df.with_columns(
        (anc.sum().over(["s1", "s3"]) - anc).alias("a_src"),
        (grp & ~pl.col("mod")).cast(pl.Int32).sum().over(["s1", "off"]).alias("g_unmod"),
        (grp & ~pl.col("s3")).cast(pl.Int32).sum().over(["s1", "off"]).alias("g2"),
        (grp & pl.col("s3")).cast(pl.Int32).sum().over(["s1", "off"]).alias("g3"))
    off12 = pl.col("off").is_in([1, 2])
    exprs = {
        "R1": off12 & (pl.col("a_src") > 0),
        "RA": off12 & pl.col("mod") & (pl.col("sib1") >= 1) & (pl.col("g_unmod") == 0),
        "RM": pl.col("ins") & pl.col("sim") & (pl.col("g2") >= 1) & (pl.col("g3") >= 1),
        "RB": pl.col("ins") & (pl.col("off") >= 3) & pl.col("mod") & (pl.col("sib1") == 0) & (pl.col("sibx") >= 1)
              & (pl.col("g_unmod") == 0),
        "W": (pl.col("xword") & (pl.col("off") >= 3))   # below 3 the mirror test says these include true typos
             | (pl.col("xstrong") & pl.col("unseen")),
        "RF": pl.col("unseen") & off12 & (pl.col("sib1") >= 1) & (pl.col("xleg") | pl.col("ndiff")) & ~pl.col("ldrop"),
        "RL": pl.col("strict") & off12 & (pl.col("sib1") >= 1) & pl.col("xleg"),
        "RW": pl.col("rw"),
    }
    df = df.with_columns([(e & pl.col("keep")).fill_null(False).alias(k) for k, e in exprs.items()])
    # R1 condemns the whole (S1, number) group of the lookalike, including its copies from the other source
    df = df.with_columns((pl.col("R1") | (pl.col("R1").any().over(["s1", "off"]) & num & pl.col("keep")))
                         .fill_null(False).alias("R1"))
    masks = {k: df[k].to_numpy() for k in exprs}
    rej = np.zeros(len(df), dtype=bool)
    for k in rules:
        rej |= masks[k]
    return rej, masks
