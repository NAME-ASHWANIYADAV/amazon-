"""Address-rendering style as a label-free tie-breaker for countries unseen in training.

The generator renders every entity's records in one style per source: the house-number format ('N° 112',
'#112', '0029', '36 - ') and whether the region is written as a region, a department or not at all. Two
confident copies of one entity agree 99% of the time; records of different entities agree at chance (3% for
the number format, 34% for the region class). So an unassigned record whose name and house number match the
S1 and whose style agrees with the S1's confident same-source copies is a missed copy (98-100% by the other,
independent style dimension), while style is left alone for rejecting (number-changed copies are re-rendered).
Style is read from the raw address text; the normalised text no longer carries it."""
import collections
import re

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein

from .normalize import _STATES

_NUMRE = re.compile(r"(?:^|,\s*)([^,\d]{0,5}?)(\d+)(\s-\s)?")
_PRE = {"": 0, "no ": 1, "n° ": 2, "#": 3, "# ": 4, "no. ": 5, "nº ": 6, "n°": 7, "nº": 8, "no.": 9}
PRES_CATS = ("EQ", "PERM", "WSUB1", "TYPO1", "NOSP", "MULTI")                 # name close to the S1's
ALL_CATS = PRES_CATS + ("DISJ", "ACRO", "MISS1", "ADD1")


def address_style(raw, country):
    """(number format code or -1, region class: 0 none / 1 region / 2 department, -1 empty)."""
    if raw is None or not raw.strip():
        return -1, -1
    m = _NUMRE.search(raw)
    if m:
        nf = _PRE.get(m.group(1).lower(), 15) * 4 + (2 if m.group(2).startswith("0") else 0) + (1 if m.group(3) else 0)
    else:
        nf = -1
    states = _STATES.get(country, {})
    canon = set(states.values())
    reg = 0
    for comp in raw.split(","):
        c = comp.strip().lower().replace("-", " ")
        if c in canon:
            reg = 1
        elif c in states:
            reg = 2
    return nf, reg


def name_category(core1, corex):
    """Edit type of the SX name core against the S1's (EQ, PERM, ADD1/2, MISS1/2, ACRO, NOSP, TYPO1, WSUB1,
    DISJ, MULTI, EMPTY)."""
    a, b = core1.split(), corex.split()
    extra = [t for t in b if t not in a]
    miss = [t for t in a if t not in b]
    if a == b:
        return "EQ"
    if not b:
        return "EMPTY"
    if sorted(a) == sorted(b):
        return "PERM"
    if not miss and extra:
        return "ADD%d" % min(len(extra), 2)
    if not extra and miss:
        return "MISS%d" % min(len(miss), 2)
    ja, jb = "".join(a), "".join(b)
    if len(b) == 1 and len(a) >= 2:
        ini = "".join(t[0] for t in a)
        if b[0] == ini or (b[0] == ini[:len(b[0])] and len(b[0]) >= 2):
            return "ACRO"
        if jb.startswith(ja[:5]) or ja.startswith(jb[:5]):
            return "NOSP"
    if len(extra) == 1 and len(miss) == 1:
        if Levenshtein.distance(extra[0], miss[0]) <= 2 or sorted(extra[0]) == sorted(miss[0]):
            return "TYPO1"
        return "WSUB1"
    if len(a) - len(miss) == 0:
        return "DISJ"
    return "MULTI"


def common_tokens(addrs, frac=0.003):
    df = collections.Counter()
    n = 0
    for a in addrs:
        if a:
            n += 1
            df.update(set(a.split()))
    return {t for t, c in df.items() if c >= frac * n}


def _rare(a, common):
    return [t for t in (a or "").split() if not t.isdigit() and t not in common and len(t) >= 3]


def _tok_match(t, toks):
    for u in toks:
        if u == t:
            return True
        if abs(len(u) - len(t)) <= 2 and min(len(u), len(t)) >= 4:
            if Levenshtein.distance(u, t, score_cutoff=2) <= 2 or sorted(u) == sorted(t):
                return True
        if len(t) >= 5 and len(u) >= 5 and (u.startswith(t) or t.startswith(u)):
            return True
    return False


def street_mismatch(a1, ax, common):
    """1 = both addresses have rare (street) tokens and none match either way; 0 = some match; -1 = undecidable."""
    r1, rx = _rare(a1, common), _rare(ax, common)
    if not r1 or not rx:
        return -1
    t1, tx = (a1 or "").split(), (ax or "").split()
    return 0 if any(_tok_match(t, tx) for t in r1) or any(_tok_match(t, t1) for t in rx) else 1


def style_additions(s1r, sxr, p, keep, pred, cand, nf, reg, src, ncat, smis, p_min=0.2):
    """Missed copies to add, by style agreement with the S1's confident same-source copies.

    All arrays per pair. keep: assignment mask; pred: current decision; cand: rows eligible as additions
    (unseen country, same house number, non-empty address, name category in ALL_CATS, SX unmatched);
    nf/reg: SX style codes; src: SX source; ncat: name category string; smis: street mismatch code.
    Anchors: kept pairs with p >= 0.99 at the same number whose name equals the S1's (EQ/PERM).
    A: number format equals an anchor's marked format and nothing contradicts;
    B: name close (PRES_CATS), region class equal to every anchor's, p >= p_min, nothing contradicts.
    One addition per SX (highest p). Returns a bool mask over pairs."""
    n = len(p)
    anchor = keep & (p >= 0.99) & np.isin(ncat, ["EQ", "PERM"]) & (nf >= -1)
    A = pl.DataFrame({"s1": s1r[anchor], "src": src[anchor], "asx": sxr[anchor], "anf": nf[anchor], "areg": reg[anchor]})
    rows = np.flatnonzero(cand & (smis != 1))
    T = pl.DataFrame({"i": rows, "s1": s1r[rows], "src": src[rows], "sx": sxr[rows], "nf": nf[rows], "reg": reg[rows]})
    J = T.join(A, on=["s1", "src"], how="inner").filter(pl.col("asx") != pl.col("sx"))
    E = J.group_by("i").agg(
        ((pl.col("areg") >= 0) & (pl.col("reg") >= 0)).sum().alias("n_rg"),
        ((pl.col("areg") >= 0) & (pl.col("reg") >= 0) & (pl.col("areg") == pl.col("reg"))).sum().alias("m_rg"),
        ((pl.col("anf") > 0) & (pl.col("nf") >= 0)).sum().alias("n_mk"),
        ((pl.col("anf") > 0) & (pl.col("nf") >= 0) & (pl.col("anf") == pl.col("nf"))).sum().alias("m_mk"),
        ((pl.col("anf") == 0) & (pl.col("nf") >= 0)).sum().alias("n_df"),
        ((pl.col("anf") == 0) & (pl.col("nf") > 0)).sum().alias("m_df_marked"))
    i = E["i"].to_numpy()
    n_rg, m_rg, n_mk, m_mk = (E[c].to_numpy() for c in ("n_rg", "m_rg", "n_mk", "m_mk"))
    n_df, m_dfm = E["n_df"].to_numpy(), E["m_df_marked"].to_numpy()
    mismatch = ((n_rg > 0) & (m_rg == 0)) | ((n_mk > 0) & (m_mk == 0)) | ((n_df > 0) & (m_dfm > 0))
    ok_a = (m_mk > 0) & ~mismatch
    ok_b = (n_rg > 0) & (m_rg == n_rg) & ~mismatch & np.isin(ncat[i], PRES_CATS) & (p[i] >= p_min)
    sel = i[ok_a | ok_b]
    if len(sel) == 0:
        return np.zeros(n, dtype=bool)
    d = pl.DataFrame({"i": sel, "sx": sxr[sel], "p": p[sel]}).sort("p", descending=True).unique("sx", keep="first")
    out = np.zeros(n, dtype=bool)
    out[d["i"].to_numpy()] = True
    return out
