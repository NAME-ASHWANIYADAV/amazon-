"""Stage 3: recalibrate the stage-2 probability with generator-structure features computed inside each S1's
candidate list, once the stage-2 probabilities identify the S1's confident copies:

- per-source address base: every source renders an entity's address from its own base, so numbers / address
  tokens that no confident same-source copy shares mark distractors (the judge underestimates this 3x);
- co-location: how many S1s share the S1's address and the SX's address (brand-only copies of a co-located
  entity look like copies of ours);
- exact duplicates among the list, and the type of name edit (typo / concatenation / novel word / replacement).

Trained on the V split (labels) on top of logit(p2); applied to test with two guards: the stage-2 p is kept
where the house number sits at a distractor shift (test fakes come in groups that imitate the per-source base)
and for countries unseen in training (no evidence of transfer)."""
import gc
import os

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

from .config import work
from .decide import assign_best
from .features import FEATURES

NUM_F = ["n_extra", "n_extra_unss", "n_extra_unall", "n_extra_osonly", "ss_conf", "all_conf"]
ADDR_F = ["n_extra_a", "n_extra_unss_a", "n_extra_unall_a", "n_extra_osonly_a"]
MISC_F = ["dup_any", "dup_conf", "addr_dup_conf", "n_s1_same_addr", "n_s1_at_sx_addr"]
EDIT_F = ["ne_novel", "ne_typo", "nm_unexpl", "ne_concat", "replace", "novel_logdf"]
NEW_F = NUM_F + ADDR_F + MISC_F + EDIT_F
XB = ["addr_empty_x", "name_inter", "name_jacc", "cos_name", "cos_addr", "hn_off", "is_s3", "legal_code",
      "comp_n_other", "comp_margin", "g_n_s1_core", "xw_n_extra", "mw_n_miss", "num_primary_eq", "hn_sib_1",
      "hn_sib_x"]
STAGE3_COLS = ["lp"] + XB + NEW_F
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4, seed=0)
ROUNDS = 400


def _tok(col):
    return pl.col(col).fill_null("").str.split(" ").list.unique()


def _sharing(base, sxs, s1s, col, suffix):
    """Extra tokens (SX tokens of `col` not in the S1's) and whether other confident copies share them."""
    xt = (base.join(sxs.select("row", col), left_on="sx", right_on="row")
          .with_columns(_tok(col).alias("tok")).drop(col).explode("tok")
          .filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")))
    s1t = (s1s.select("row", col).with_columns(_tok(col).alias("tok")).drop(col).explode("tok")
           .filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")).rename({"row": "s1"})
           .with_columns(pl.lit(True).alias("in1")))
    xt = xt.join(s1t, on=["s1", "tok"], how="left").with_columns(pl.col("in1").fill_null(False))
    c = pl.col("cne").cast(pl.Int32)
    xt = xt.with_columns((c.sum().over(["s1", "src", "tok"]) - c).alias("ss_tok"),
                         (c.sum().over(["s1", "tok"]) - c).alias("all_tok"))
    ext = xt.filter(~pl.col("in1"))
    return ext.group_by("i").agg(
        pl.len().alias("n_extra" + suffix),
        (pl.col("ss_tok") == 0).sum().alias("n_extra_unss" + suffix),
        (pl.col("all_tok") == 0).sum().alias("n_extra_unall" + suffix),
        ((pl.col("ss_tok") == 0) & (pl.col("all_tok") > 0)).sum().alias("n_extra_osonly" + suffix))


def _name_edit(s1_names, sx_names, df):
    """df: token -> number of SX records in the whole split whose name_core contains it."""
    out = np.zeros((len(s1_names), 6), dtype=np.float32)
    for k, (a, b) in enumerate(zip(s1_names, sx_names)):
        if not a or not b:
            continue
        A, B = set(a.split()), set(b.split())
        E, M = B - A, A - B
        if not E and not M:
            continue
        ne = nt = nc = 0
        mdf = 99.0
        matched = set()
        ajoin = "".join(a.split())
        for e in E:
            best, bm = 0.0, None
            for m in M:
                s = Levenshtein.normalized_similarity(e, m)
                if s > best:
                    best, bm = s, m
            if best >= 0.6:
                nt += 1
                matched.add(bm)
            elif len(e) >= 4 and e in ajoin:
                nc += 1
            else:
                ne += 1
                mdf = min(mdf, float(np.log1p(df.get(e, 0))))
        nm = len(M - matched)
        out[k] = (ne, nt, nm, nc, 1.0 if (ne > 0 and nm > 0) else 0.0, mdf if ne else -1.0)
    return out


def build_features(d, sxs, s1s, s1_addr_cnt, tok_df):
    """d: i, s1, sx, p, keep (keep = assign_best over the whole split); sxs / s1s: the prep rows these pairs
    use (with a 'row' column); s1_addr_cnt: ha -> number of S1s at that address; tok_df: tok -> SX count."""
    sxs = sxs.with_columns(pl.col("row").cast(pl.Int64))
    s1s = s1s.with_columns(pl.col("row").cast(pl.Int64))
    d = d.with_columns(pl.col("s1").cast(pl.Int64), pl.col("sx").cast(pl.Int64))
    d = d.join(sxs.select("row", "src", "addr_empty"), left_on="sx", right_on="row", how="left")
    conf = (pl.col("p") >= 0.5) & pl.col("keep")
    d = d.with_columns((conf & ~pl.col("addr_empty")).alias("cne"), conf.alias("conf2"))
    base = d.select("i", "s1", "sx", "src", "cne")
    fn = _sharing(base, sxs, s1s, "numbers", "")
    fa = _sharing(base, sxs, s1s, "addr_norm", "_a")
    c = pl.col("cne").cast(pl.Int32)
    d = d.with_columns((c.sum().over(["s1", "src"]) - c).alias("ss_conf"), (c.sum().over("s1") - c).alias("all_conf"))
    h = sxs.select("row", (pl.col("name_norm").fill_null("") + "|" + pl.col("addr_norm").fill_null("")).hash().alias("hna"),
                   pl.col("addr_norm").fill_null("").hash().alias("ha"))
    d = d.join(h, left_on="sx", right_on="row", how="left")
    c2 = pl.col("conf2").cast(pl.Int32)
    d = d.with_columns((pl.len().over(["s1", "hna"]) - 1).alias("dup_any"),
                       (c2.sum().over(["s1", "hna"]) - c2).alias("dup_conf"),
                       (c2.sum().over(["s1", "ha"]) - c2).alias("addr_dup_conf"))
    s1h = s1s.select(pl.col("row").alias("s1"), pl.col("addr_norm").fill_null("").hash().alias("ha1"))
    d = d.join(s1h, on="s1", how="left")
    d = d.join(s1_addr_cnt.rename({"ha": "ha1", "n1": "n_s1_same_addr"}), on="ha1", how="left")
    d = d.join(s1_addr_cnt.rename({"n1": "n_s1_at_sx_addr"}), on="ha", how="left")
    d = d.join(fn, on="i", how="left").join(fa, on="i", how="left").sort("i")
    d = d.with_columns([pl.col(k).fill_null(0) for k in NUM_F + ADDR_F + ["n_s1_same_addr", "n_s1_at_sx_addr"]])
    nm1 = dict(zip(s1s["row"].to_list(), s1s["name_core"].to_list()))
    nmx = dict(zip(sxs["row"].to_list(), sxs["name_core"].to_list()))
    toks = sxs.select(_tok("name_core").alias("tok")).explode("tok").unique()
    sub = toks.join(tok_df, on="tok", how="inner")
    dfd = dict(zip(sub["tok"].to_list(), sub["cnt"].to_list()))
    ed = _name_edit([nm1[a] for a in d["s1"].to_list()], [nmx[b] for b in d["sx"].to_list()], dfd)
    d = d.with_columns([pl.Series(k, ed[:, j]) for j, k in enumerate(EDIT_F)])
    return d.select(["i"] + NEW_F).with_columns([pl.col(k).cast(pl.Float32) for k in NEW_F])


def s1_addr_counts(s1_prep_path):
    a = pl.read_parquet(s1_prep_path, columns=["addr_norm"])
    return a.select(pl.col("addr_norm").fill_null("").hash().alias("ha")).group_by("ha").agg(pl.len().alias("n1"))


def token_df(sx_prep_path):
    """Per name_core token: number of SX records of the split containing it (streamed in batches)."""
    acc = None
    for batch in pq.ParquetFile(sx_prep_path).iter_batches(batch_size=1_000_000, columns=["name_core"]):
        t = (pl.from_arrow(batch).select(_tok("name_core").alias("tok")).explode("tok")
             .filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")).group_by("tok").agg(pl.len().alias("cnt")))
        acc = t if acc is None else pl.concat([acc, t]).group_by("tok").agg(pl.col("cnt").sum())
    return acc.with_columns(pl.col("cnt").cast(pl.Int64))


def split_features(prep, s1r, sxr, p, out_path, per_chunk=800_000, log=print):
    """NEW_F for all pairs (grouped by s1) of a split, in S1 chunks (bounded memory); writes out_path (i + NEW_F)."""
    n = len(p)
    keep = assign_best(sxr, p)
    assert np.all(np.diff(s1r) >= 0), "pairs must be grouped by s1 (sorted)"
    cnt = s1_addr_counts(work("prep", f"{prep}_s1.parquet"))
    tdf = token_df(work("prep", f"{prep}_sx.parquet"))
    starts = np.flatnonzero(np.r_[True, s1r[1:] != s1r[:-1]])
    bounds = [0]
    for b in starts:
        if b - bounds[-1] >= per_chunk:
            bounds.append(int(b))
    bounds.append(n)
    s1p = work("prep", f"{prep}_s1.parquet")
    pf = pq.ParquetFile(work("prep", f"{prep}_sx.parquet"), pre_buffer=False)
    cols = ["name_norm", "addr_norm", "name_core", "numbers", "addr_empty", "src"]
    parts = []
    for c in range(len(bounds) - 1):
        a, b = bounds[c], bounds[c + 1]
        mask = np.zeros(pf.metadata.num_rows, dtype=bool)
        mask[np.unique(sxr[a:b])] = True
        got, off = [], 0
        for batch in pf.iter_batches(batch_size=122_880, columns=cols, use_threads=False):
            m = mask[off:off + batch.num_rows]
            if m.any():
                got.append(batch.filter(pa.array(m)).append_column(
                    "row", pa.array(np.flatnonzero(m).astype(np.int64) + off)))
            off += batch.num_rows
        sxs = pl.from_arrow(pa.Table.from_batches(got))
        del got, mask
        d = pl.DataFrame({"i": np.arange(a, b, dtype=np.int64), "s1": s1r[a:b].astype(np.int64),
                          "sx": sxr[a:b].astype(np.int64), "p": p[a:b], "keep": keep[a:b]})
        lo, hi = int(s1r[a]), int(s1r[b - 1])
        s1c = (pl.scan_parquet(s1p).select(["name_core", "addr_norm", "numbers"]).slice(lo, hi - lo + 1).collect()
               .with_row_index("row", offset=lo)
               .filter(pl.col("row").is_in(pl.Series(np.unique(s1r[a:b]).astype(np.uint32)).implode())))
        part = out_path + f".part{c:03d}"
        build_features(d, sxs, s1c, cnt, tdf).write_parquet(part)
        parts.append(part)
        del sxs, d, s1c
        gc.collect()
        log(f"  stage-3 features chunk {c + 1}/{len(bounds) - 1} (rows {a}-{b})")
    pl.scan_parquet(parts).sink_parquet(out_path)
    for q in parts:
        os.remove(q)


def _logit(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


def matrix(X, feats, p, a, b, f0=0):
    """float32 (b-a, len(STAGE3_COLS)) for pair rows [a, b): logit(p2), XB columns of X, NEW_F from feats
    (feats arrays start at pair row f0)."""
    idx = [FEATURES.index(c) for c in XB]
    M = np.empty((b - a, len(STAGE3_COLS)), dtype=np.float32)
    M[:, 0] = _logit(p[a:b])
    for s in range(a, b, 200_000):
        e = min(b, s + 200_000)
        M[s - a:e - a, 1:1 + len(XB)] = X[s:e][:, idx]
    for j, c in enumerate(NEW_F):
        M[:, 1 + len(XB) + j] = feats[c][a - f0:b - f0]
    return M


def train(X, feats, p, y):
    return lgb.train(PARAMS, lgb.Dataset(matrix(X, feats, p, 0, len(p)), y.astype(np.float32)), num_boost_round=ROUNDS)


def predict(model, X, feats_path, p, chunk=1_000_000):
    out = np.empty(len(p), dtype=np.float32)
    for a in range(0, len(p), chunk):
        b = min(len(p), a + chunk)
        f = pl.scan_parquet(feats_path).slice(a, b - a).collect()
        assert f["i"][0] == a and f.height == b - a
        feats = {c: f[c].to_numpy() for c in NEW_F}
        out[a:b] = model.predict(matrix(X, feats, p, a, b, f0=a))
        del f, feats
    return out


def guard(p2, p3, hn_in_set, unseen):
    """Stage-2 p where the SX sits at a distractor shift (min of both) or the country was never trained on."""
    p = np.where(hn_in_set, np.minimum(p2, p3), p3)
    return np.where(unseen, p2, p).astype(np.float32)
