"""CLI for every pipeline stage: python -m src.run_pipeline <stage> [options]."""
import argparse
import gc
import os
import time
import zlib
from multiprocessing import Pool

import numpy as np
import polars as pl

from . import config
from .config import work
from .io_utils import read_gt_pairs, read_source
from .normalize import normalize_rows

PREP_COLS = ["name_norm", "name_core", "legal", "alt_core", "was_indic", "is_domain",
             "addr_norm", "numbers", "addr_empty"]
BOOL_COLS = ("was_indic", "is_domain", "addr_empty")
CHUNK = 100_000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def part_of(entity_id):
    h = zlib.crc32(entity_id.encode()) % 100
    return "E" if h < config.SPLIT_E else ("J" if h < config.SPLIT_J else "V")


def load_prep(split, which, columns=None):
    return pl.read_parquet(work("prep", f"{split}_{which}.parquet"), columns=columns)


def _normalize_frame(df, pool):
    rows = list(zip(df["business_name"].to_list(), df["business_address"].to_list(),
                    df["country"].to_list()))
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    schema = [(c, pl.Boolean if c in BOOL_COLS else pl.Utf8) for c in PREP_COLS]
    parts = [pl.DataFrame(res, schema=schema, orient="row") for res in pool.imap(normalize_rows, chunks)]
    return pl.concat([df.select("entity_id", "country"), pl.concat(parts)], how="horizontal")


def stage_prepare(args):
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        for split in ("train", "test"):
            s1 = _normalize_frame(read_source(split, 1), pool)
            if split == "train":
                s1 = s1.with_columns(pl.Series("part", [part_of(e) for e in s1["entity_id"].to_list()]))
            s1.write_parquet(work("prep", f"{split}_s1.parquet"))
            log(split, "s1", s1.shape)
            sx = pl.concat([_normalize_frame(read_source(split, s), pool)
                            .with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (2, 3)])
            sx.write_parquet(work("prep", f"{split}_sx.parquet"))
            log(split, "sx", sx.shape)
    s1 = load_prep("train", "s1", ["entity_id"]).with_row_index("s1_row")
    sx = load_prep("train", "sx", ["entity_id"]).with_row_index("sx_row")
    gt = (read_gt_pairs().join(s1, left_on="s1", right_on="entity_id")
          .join(sx, left_on="sx", right_on="entity_id")
          .select(pl.col("s1_row").cast(pl.Int32), pl.col("sx_row").cast(pl.Int32)))
    gt.write_parquet(work("prep", "train_gt_rows.parquet"))
    log("gt rows", gt.shape)


def stage_train_encoder(args):
    import torch

    from .encoder import train_encoder
    s1 = load_prep("train", "s1", ["name_norm", "addr_norm", "part"])
    sx = load_prep("train", "sx", ["name_norm", "addr_norm"])
    gt = pl.read_parquet(work("prep", "train_gt_rows.parquet"))
    is_e = (s1["part"] == "E").to_numpy()
    gt = gt.filter(pl.Series(is_e[gt["s1_row"].to_numpy()]))
    log("encoder pairs (E)", gt.height)
    model = train_encoder(s1["name_norm"], s1["addr_norm"], sx["name_norm"], sx["addr_norm"],
                          gt["s1_row"].to_numpy().astype(np.int64), gt["sx_row"].to_numpy().astype(np.int64),
                          epochs=args.epochs, log=log)
    torch.save(model.state_dict(), work("models", "encoder.pt"))


def load_encoder():
    import torch

    from .encoder import DEVICE, FingerprintModel
    model = FingerprintModel().to(DEVICE)
    model.load_state_dict(torch.load(work("models", "encoder.pt"), map_location=DEVICE, weights_only=True))
    return model


def stage_encode(args):
    from .encoder import encode_to
    model = load_encoder()
    for which in ("s1", "sx"):
        df = load_prep(args.split, which, ["name_norm", "addr_norm"])
        encode_to(model, df["name_norm"], df["addr_norm"], work("emb", f"{args.split}_{which}.npy"))
        log("encoded", args.split, which, df.height)


def stage_candidates(args):
    from .candidates import build_candidates, recall_report
    s1 = load_prep(args.split, "s1", ["country"] + (["part"] if args.split == "train" else []))
    sx = load_prep(args.split, "sx", ["country", "addr_empty"])
    s1_emb = np.load(work("emb", f"{args.split}_s1.npy"), mmap_mode="r")
    sx_emb = np.load(work("emb", f"{args.split}_sx.npy"), mmap_mode="r")
    name = args.split
    if args.split == "train":
        is_e = (s1["part"] == "E").to_numpy()
        q_rows = np.flatnonzero(is_e if args.queries == "E" else ~is_e)
        if args.queries == "E":
            _competitor_lists_e(s1, sx, s1_emb, sx_emb, q_rows)
            return
    else:
        q_rows = np.arange(s1.height)
    cand = build_candidates(s1["country"].to_numpy(), sx["country"].to_numpy(), q_rows, s1_emb, sx_emb,
                            config.TOP_K, sx_addr_empty=sx["addr_empty"].to_numpy(), k_addr=args.k_addr, log=log)
    if name == "train_E":
        cand = cand.select("s1", "sx", "cos", "cos_name")
    cand.write_parquet(work("cand", f"{name}.parquet"))
    log("candidates", name, cand.shape)
    if name == "train":
        gt = pl.read_parquet(work("prep", "train_gt_rows.parquet"))
        for p in ("J", "V"):
            log(f"recall on {p}:")
            recall_report(cand, gt, np.flatnonzero((s1["part"] == p).to_numpy()), log=log)


def _competitor_lists_e(s1, sx, s1_emb, sx_emb, q_rows):
    """Top-COMPETITOR_RANK record neighbours of the E-split S1s (competitors only). Each country is written
    to disk as soon as it is done, and no global sort/unique is needed, so memory stays small."""
    from .candidates import _pairs_frame, knn
    from .features import COMPETITOR_RANK
    s1c, sxc, empty = s1["country"].to_numpy(), sx["country"].to_numpy(), sx["addr_empty"].to_numpy()
    paths = []
    for c in sorted(set(s1c[q_rows].tolist())):
        qr = q_rows[s1c[q_rows] == c]
        dr = np.flatnonzero(sxc == c)
        log(f"knn E {c}: {len(qr)} queries x {len(dr)} records, top-{COMPETITOR_RANK}")
        res = knn(s1_emb[qr], sx_emb[dr], COMPETITOR_RANK, d_scale=np.where(empty[dr], np.sqrt(2.0), 1.0))
        path = work("cand", f"train_E_{c}.parquet")
        _pairs_frame(qr, dr, *res).select("s1", "sx", "cos", "cos_name").write_parquet(path)
        paths.append(path)
        del res
        gc.collect()
    e = pl.concat([pl.read_parquet(p) for p in paths])
    e.write_parquet(work("cand", "train_E.parquet"))
    for p in paths:
        os.remove(p)
    log("candidates train_E", e.shape)


def stage_address_pass(args):
    """Add top-k_addr address-cosine neighbours to an existing candidates file (in place, backup kept)."""
    import shutil

    from .candidates import add_address_pass, recall_report
    s1 = load_prep(args.split, "s1", ["country"] + (["part"] if args.split == "train" else []))
    sx = load_prep(args.split, "sx", ["country", "addr_empty"])
    path = work("cand", f"{args.split}.parquet")
    backup = work("cand", f"{args.split}_record_only.parquet")
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
    parts = []

    def sink(c, frame):  # one merged file per country, combined at the end (bounded memory)
        parts.append(work("cand", f"{args.split}_merged_{c}.parquet"))
        frame.write_parquet(parts[-1])

    knn_file = lambda c: work("cand", f"{args.split}_addrknn{args.k_addr}_{c}.parquet")  # kNN checkpoints
    add_address_pass(pl.read_parquet(backup, columns=["s1", "sx", "cos", "cos_name", "cos_addr"]),
                     s1["country"].to_numpy(), sx["country"].to_numpy(),
                     np.load(work("emb", f"{args.split}_s1.npy"), mmap_mode="r"),
                     np.load(work("emb", f"{args.split}_sx.npy"), mmap_mode="r"),
                     args.k_addr, sx_addr_empty=sx["addr_empty"].to_numpy(), log=log, checkpoint=knn_file,
                     sink=sink)
    gc.collect()
    cand = pl.concat([pl.read_parquet(f) for f in parts], rechunk=False)
    cand.write_parquet(path)
    log("candidates with address pass", cand.shape)
    for f in parts:
        os.remove(f)
    for c in np.unique(s1["country"].to_numpy()):
        if os.path.exists(knn_file(c)):
            os.remove(knn_file(c))
    if args.split == "train":
        gt = pl.read_parquet(work("prep", "train_gt_rows.parquet"))
        log("recall on V:")
        recall_report(cand, gt, np.flatnonzero((s1["part"] == "V").to_numpy()), log=log)


FEAT_CHUNK = 2_000_000


def _load_tok(split):
    """Token arrays of a split plus per-record country codes (cc_s1, cc_sx), which make idf and global name
    counts per-country and size-normalised (features.country_idf, context.global_name_counts)."""
    from .features import COUNTRY_CODES, load_token_arrays
    tok = load_token_arrays(work("tok", f"{split}.npz"))
    for which in ("s1", "sx"):
        tok[f"cc_{which}"] = (load_prep(split, which, ["country"])["country"]
                              .replace_strict(COUNTRY_CODES, default=3, return_dtype=pl.Int8).to_numpy())
    return tok


def _load_split_frames(split):
    """Only the string columns the featurizer reads directly (everything else is in the token arrays)."""
    s1 = load_prep(split, "s1", ["name_core", "addr_norm", "entity_id", "country"]
                   + (["part"] if split == "train" else []))
    sx = load_prep(split, "sx", ["name_core", "alt_core", "addr_norm", "entity_id"])
    return s1, sx


def stage_tokens(args):
    """Token/number/legal/flag arrays for a split, one column at a time (bounded memory)."""
    from .features import build_token_arrays

    def load_col(which, col):
        s = load_prep(args.split, which, [col])[col]
        return s.to_list() if s.dtype == pl.Utf8 else s.to_numpy()

    tok = build_token_arrays(load_col, log=log)
    np.savez(work("tok", f"{args.split}.npz"), **tok)
    log("tokens saved", args.split, {k: v.shape for k, v in tok.items()})


def _featurizer(split):
    from .features import PairFeaturizer, load_token_arrays
    s1, sx = _load_split_frames(split)
    return s1, sx, PairFeaturizer(s1, sx, _load_tok(split))


def _pruned_candidates(split, floor, with_comp=True, dropped=None):
    """Candidates with cos >= floor, plus per-S1 list features computed on the pruned lists and cross-S1
    competition features. For train the competitor lists of the E-split S1s (cand/train_E.parquet) are
    included so that every SX sees all its competing S1s, as it does on test.
    dropped (bool per S1 row, optional): these S1s vanish as queries and as competitors, so their SX become
    orphans the way SX of S1s missing from the test file are."""
    from .features import add_list_features
    cand = pl.read_parquet(work("cand", f"{split}.parquet"))
    if floor is not None:
        cand = cand.filter(pl.col("cos") >= floor)
    if dropped is not None:
        cand = cand.filter(pl.Series(~dropped[cand["s1"].to_numpy()]))
    cand = add_list_features(cand)
    if not with_comp:
        return cand
    extra = [pl.read_parquet(work("cand", "train_E.parquet"), columns=["s1", "sx", "cos", "cos_name"])] \
        if split == "train" else []
    if dropped is not None:
        extra = [e.filter(pl.Series(~dropped[e["s1"].to_numpy()])) for e in extra]
    return _with_competition(cand, extra)


def _dropped_s1(n_s1):
    """The simulated-missing S1 rows of train (models/dropped_s1.npy, written by `features --drop-frac`)."""
    f = work("models", "dropped_s1.npy")
    return np.load(f) if os.path.isfile(f) else np.zeros(n_s1, dtype=bool)


def _with_competition(cand, extra_competitors=()):
    """Add COMP_COLS: competitors are every S1's top-COMPETITOR_RANK record candidates (rank < 20 in these
    lists, plus any extra competitor lists such as the E-split top-20), identical for train and test."""
    from .features import COMP_COLS, COMPETITOR_RANK, competition_vs_table
    t = pl.concat([cand.filter(pl.col("rank") < COMPETITOR_RANK).select("s1", "sx", "cos", "cos_name")]
                  + [e.select("s1", "sx", "cos", "cos_name") for e in extra_competitors])
    comp = competition_vs_table(cand["s1"].to_numpy(), cand["sx"].to_numpy(), cand["cos"].to_numpy(),
                                cand["cos_name"].to_numpy(), t["s1"].to_numpy(), t["sx"].to_numpy(),
                                t["cos"].to_numpy(), t["cos_name"].to_numpy())
    del t
    gc.collect()
    return cand.with_columns([pl.Series(c, comp[:, i]) for i, c in enumerate(COMP_COLS)])


def _saved_floor():
    import json
    with open(work("models", "floor.json")) as f:
        return json.load(f)["floor"]


def _with_addr_empty(cand, tok):
    return cand.with_columns(pl.Series("addr_empty_x", tok["flags_sx"][cand["sx"].to_numpy(), 2]))


def _stage_a_matrix(cand):
    from .features import STAGE_A_COLS
    return np.column_stack([cand[k].to_numpy().astype(np.float32) for k in STAGE_A_COLS])


def _stage_a_predict(cand, models, chunk=4_000_000):
    """Mean stage-A probability of the given models, built chunk by chunk from the table columns."""
    from . import judge
    out = np.empty(cand.height, dtype=np.float32)
    for b in range(0, cand.height, chunk):
        X = _stage_a_matrix(cand.slice(b, chunk))
        out[b:b + chunk] = np.mean([judge.predict(m, X) for m in models], axis=0)
    return out


def _add_context(cand, tok, lo_extra, lo_miss, s1_present=None):
    """Generator-aware context columns (rows must be grouped by s1)."""
    from .context import CTX_COLS, context_features
    F = context_features(tok, cand["s1"].to_numpy(), cand["sx"].to_numpy(), cand["cos_name"].to_numpy(),
                         cand["cos_addr"].to_numpy(), lo_extra, lo_miss, s1_present)
    return cand.with_columns([pl.Series(c, F[:, i]) for i, c in enumerate(CTX_COLS)])


def stage_vocab(args):
    """Name-token vocabulary in token-id order (needed to carry word log-odds from train to test ids)."""
    from .features import build_vocab
    vocab = build_vocab(load_prep(args.split, w, ["name_core"])["name_core"].to_list() for w in ("s1", "sx"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    pl.DataFrame({"token": tokens}).write_parquet(work("tok", f"{args.split}_vocab_n.parquet"))
    log("vocab", args.split, len(tokens))


def stage_features(args):
    import json

    from . import judge
    from .context import word_log_odds
    from .features import FEATURES, PairFeaturizer, load_token_arrays
    with open(work("models", "floor.json"), "w") as f:
        json.dump({"floor": args.floor}, f)
    s1, sx = _load_split_frames("train")
    tok = _load_tok("train")
    fz = PairFeaturizer(s1, sx, tok)
    dropped = np.random.default_rng(20260927).random(s1.height) < args.drop_frac
    if args.drop_frac > 0:
        np.save(work("models", "dropped_s1.npy"), dropped)
        log(f"simulated missing S1s: {int(dropped.sum())} of {s1.height}")
    elif os.path.isfile(work("models", "dropped_s1.npy")):
        os.remove(work("models", "dropped_s1.npy"))
    present = ~dropped
    cand = _pruned_candidates("train", args.floor, dropped=dropped if args.drop_frac > 0 else None)
    part = s1["part"].to_numpy()
    gt = pl.read_parquet(work("prep", "train_gt_rows.parquet")).with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    cand = (cand.join(gt.rename({"s1_row": "s1", "sx_row": "sx"}), on=["s1", "sx"], how="left")
            .with_columns(pl.col("label").fill_null(0)).sort("s1", "rank"))
    cand = _with_addr_empty(cand, tok)
    log("candidates J+V", cand.height)

    # ---- stage A: cheap filter on list/competition features, 2-fold out-of-fold on J
    s1r, y = cand["s1"].to_numpy(), cand["label"].to_numpy()
    is_j, fold = part[s1r] == "J", s1r % 2
    models = []
    for f in (0, 1):
        tr = is_j & (fold == f)
        models.append(judge.train_small(_stage_a_matrix(cand.filter(pl.Series(tr))), y[tr]))
        models[-1].save_model(work("models", f"stage_a_{f}.json"))
    pA = np.empty(cand.height, dtype=np.float32)
    for f in (0, 1):  # model f scores the other fold
        te = is_j & (fold != f)
        pA[te] = _stage_a_predict(cand.filter(pl.Series(te)), [models[f]])
    pA[~is_j] = _stage_a_predict(cand.filter(pl.Series(~is_j)), models)
    t_a = float(np.quantile(pA[is_j & (y == 1)], 0.001))
    with open(work("models", "stage_a.json"), "w") as f:
        json.dump({"threshold": t_a}, f)
    keep = pA >= t_a
    for p in ("J", "V"):
        m = part[s1r] == p
        log(f"stage A {p}: kept {keep[m].sum()} of {m.sum()} pairs ({keep[m].sum() / len(np.unique(s1r[m])):.1f}/S1), "
            f"true kept {(keep & m & (y == 1)).sum() / max(1, (m & (y == 1)).sum()):.4f}, threshold {t_a:.5f}")
    cand = cand.with_columns(pl.Series("pA", pA)).filter(pl.Series(keep))
    del pA, keep, s1r, y
    gc.collect()

    # ---- word log-odds (out-of-fold for J) and generator-aware context features
    s1r, sxr, y = cand["s1"].to_numpy(), cand["sx"].to_numpy(), cand["label"].to_numpy()
    is_j, fold = part[s1r] == "J", s1r % 2
    lo = {f: word_log_odds(tok, s1r[is_j & (fold == f)], sxr[is_j & (fold == f)], y[is_j & (fold == f)])
          for f in (0, 1)}
    lo_full = word_log_odds(tok, s1r[is_j], sxr[is_j], y[is_j])
    np.savez(work("models", "word_lo_train.npz"), extra=lo_full[0], miss=lo_full[1])
    pieces = [_add_context(cand.filter(pl.Series(is_j & (fold == 0))), tok, *lo[1], present),
              _add_context(cand.filter(pl.Series(is_j & (fold == 1))), tok, *lo[0], present),
              _add_context(cand.filter(pl.Series(~is_j)), tok, *lo_full, present)]
    cand = pl.concat(pieces)
    del pieces
    gc.collect()
    log("featurizer ready")
    for p in ("J", "V"):
        c = cand.filter(pl.Series(part[cand["s1"].to_numpy()] == p))
        if p == "J":  # early-stopping slice (S1 row % 10 == 0) goes last
            c = c.with_columns((pl.col("s1") % 10 == 0).alias("ev")).sort("ev", "s1", "rank").drop("ev")
        out = np.lib.format.open_memmap(work("feat", f"train_{p}.npy"), mode="w+", dtype=np.float16,
                                        shape=(c.height, len(FEATURES)))  # float16 halves disk use
        for b in range(0, c.height, FEAT_CHUNK):
            out[b:b + FEAT_CHUNK] = fz.compute(c.slice(b, FEAT_CHUNK))
            log(f"  {p} features {min(b + FEAT_CHUNK, c.height)}/{c.height}")
        out.flush()
        del out
        c.select("s1", "sx", "label").write_parquet(work("feat", f"train_{p}_pairs.parquet"))
        log(p, "pairs", c.height, "positives", int(c["label"].sum()))


def stage_train_judge(args):
    from . import judge
    X = np.load(work("feat", "train_J.npy")).astype(np.float32)
    pairs = pl.read_parquet(work("feat", "train_J_pairs.parquet"))
    y = pairs["label"].to_numpy().astype(np.float32)
    n_eval = int((pairs["s1"] % 10 == 0).sum())
    booster = judge.train(X, y, n_eval)
    booster.save_model(work("models", "judge_v1.json"))
    log("judge best iteration", booster.best_iteration)


def _truth_by_s1(s1_rows, s1_ids, sx_ids):
    gt = pl.read_parquet(work("prep", "train_gt_rows.parquet")).filter(
        pl.col("s1_row").is_in(pl.Series(np.asarray(s1_rows, dtype=np.int32)).implode()))
    truth = {s1_ids[r]: set() for r in s1_rows}
    for a, b in zip(gt["s1_row"].to_list(), gt["sx_row"].to_list()):
        truth[s1_ids[a]].add(sx_ids[b])
    return truth


def _score(s1r, sxr, mask, s1_rows, s1_ids, sx_ids, truth):
    from .evaluate import macro_f05
    pred = {s1_ids[r]: set() for r in s1_rows}
    for a, b in zip(s1r[mask].tolist(), sxr[mask].tolist()):
        pred[s1_ids[a]].add(sx_ids[b])
    return macro_f05(pred, truth, [s1_ids[r] for r in s1_rows])


def stage_validate(args):
    import xgboost as xgb

    from . import judge
    booster = xgb.Booster(model_file=work("models", "judge_v1.json"))
    p = judge.predict(booster, np.load(work("feat", "train_V.npy"), mmap_mode="r"))
    np.save(work("pred", "train_V_p.npy"), p)
    _validate_probs(p, "decision.json")


def _validate_probs(p, decision_file):
    """Macro F0.5 on V (and the stress view) for pair probabilities p aligned with train_V_pairs; picks the
    decision rule and writes it to models/<decision_file>."""
    import json

    from .decide import assign_best, expected_f05_cut, threshold_cut
    s1 = load_prep("train", "s1", ["entity_id", "country", "part"])
    sx_ids = load_prep("train", "sx", ["entity_id"])["entity_id"].to_numpy()
    s1_ids = s1["entity_id"].to_numpy()
    country = s1["country"].to_numpy()
    pairs = pl.read_parquet(work("feat", "train_V_pairs.parquet"))
    s1r, sxr = pairs["s1"].to_numpy(), pairs["sx"].to_numpy()
    rng = np.random.default_rng(0)
    v_rows = np.flatnonzero((s1["part"] == "V").to_numpy() & ~_dropped_s1(s1.height))
    views = {"V": v_rows, "V-stress": v_rows[rng.random(len(v_rows)) >= 0.19]}
    results = {}
    for view, rows in views.items():
        truth = _truth_by_s1(rows, s1_ids, sx_ids)
        inview = np.isin(s1r, rows)
        a, b, pp = s1r[inview], sxr[inview], p[inview]
        keep = assign_best(b, pp)
        a, b, pp = a[keep], b[keep], pp[keep]
        res = {f"thr{t:.2f}": _score(a, b, threshold_cut(pp, t), rows, s1_ids, sx_ids, truth)
               for t in np.arange(0.2, 0.91, 0.05)}
        res["expected_f"] = _score(a, b, expected_f05_cut(a, pp), rows, s1_ids, sx_ids, truth)
        for c in sorted(set(country[rows].tolist())):
            crow = rows[country[rows] == c]
            cset = set(s1_ids[crow].tolist())
            ct = {k: v for k, v in truth.items() if k in cset}
            m = np.isin(a, crow)
            res[f"expected_f[{c}]"] = _score(a[m], b[m], expected_f05_cut(a[m], pp[m]), crow, s1_ids, sx_ids, ct)
        results[view] = res
        best = max((k for k in res if k.startswith("thr")), key=res.get)
        log(view, "best threshold", best, round(res[best], 5), "| expected_f", round(res["expected_f"], 5))
        log(view, {k: round(v, 4) for k, v in res.items()})
    thr_key = max((k for k in results["V"] if k.startswith("thr")),
                  key=lambda k: results["V"][k] + results["V-stress"][k])
    use_ef = (results["V"]["expected_f"] + results["V-stress"]["expected_f"] >=
              results["V"][thr_key] + results["V-stress"][thr_key])
    decision = {"rule": "expected_f" if use_ef else "threshold", "threshold": float(thr_key[3:]),
                "floor": _saved_floor(), "v_score": results["V"]["expected_f" if use_ef else thr_key]}
    with open(work("models", decision_file), "w") as f:
        json.dump(decision, f)
    log("decision", decision)
    return decision


def _stage2_models_and_probs(XJ, pj, XV, pv, tok):
    """Stage 2: out-of-fold stage-1 probabilities on J (2 folds by s1), stage-2 features, stage-2 judge.
    Returns the stage-2 V probabilities; models are saved under models/."""
    from . import judge
    from .context import stage2_features
    s1j, yj = pj["s1"].to_numpy(), pj["label"].to_numpy().astype(np.float32)
    ev = (s1j % 10 == 0)
    fold = s1j % 2
    p1j = np.empty(len(yj), dtype=np.float32)
    fold_models = []
    for f in (0, 1):
        idx = np.flatnonzero(fold == f)
        ev_f = (s1j[idx] // 2) % 10 == 0  # early-stopping slice inside the fold (s1 % 10 is always even)
        idx = np.r_[idx[~ev_f], idx[ev_f]]
        m = judge.train(XJ[idx], yj[idx], int(ev_f.sum()))
        m.save_model(work("models", f"judge_f{f}.json"))
        fold_models.append(m)
        p1j[fold != f] = judge.predict(m, XJ[fold != f])
        log(f"stage-1 fold {f}: best iteration {m.best_iteration}")
    p1v = np.mean([judge.predict(m, XV) for m in fold_models], axis=0)
    S2j = stage2_features(tok, s1j, pj["sx"].to_numpy(), p1j)
    S2v = stage2_features(tok, pv["s1"].to_numpy(), pv["sx"].to_numpy(), p1v)
    m2 = judge.train(np.hstack([XJ, S2j]), yj, int(ev.sum()))
    m2.save_model(work("models", "judge_s2.json"))
    log("stage-2 best iteration", m2.best_iteration)
    return judge.predict(m2, np.hstack([XV, S2v])), p1v


def stage_stage2(args):
    """Train the stacked stage-2 judge on J and validate it on V (writes models/decision_s2.json)."""
    from .features import load_token_arrays
    tok = _load_tok("train")
    XJ = np.load(work("feat", "train_J.npy")).astype(np.float32)
    XV = np.load(work("feat", "train_V.npy")).astype(np.float32)
    pj = pl.read_parquet(work("feat", "train_J_pairs.parquet"))
    pv = pl.read_parquet(work("feat", "train_V_pairs.parquet"))
    p2, p1v = _stage2_models_and_probs(XJ, pj, XV, pv, tok)
    np.save(work("pred", "train_V_p2.npy"), p2)
    log("stage-1 (OOF-fold average) on V:")
    _validate_probs(p1v, "decision_s1avg.json")
    log("stage-2 on V:")
    _validate_probs(p2, "decision_s2.json")


def stage_stage2_final(args):
    """Final models: the stage-2 fold judges and the stage-2 judge retrained on J + V together (66% more data;
    halving J cost 0.00074 on V, so more data helps) with fixed round counts, since no data is left to early-stop
    on. decision_s2.json from the validated run is kept. Overwrites judge_f0/judge_f1/judge_s2."""
    import xgboost as xgb

    from . import judge
    from .context import stage2_features
    tok = _load_tok("train")
    pj = pl.read_parquet(work("feat", "train_J_pairs.parquet"))
    pv = pl.read_parquet(work("feat", "train_V_pairs.parquet"))
    X = np.concatenate([np.load(work("feat", "train_J.npy")), np.load(work("feat", "train_V.npy"))]).astype(np.float32)
    s1 = np.r_[pj["s1"].to_numpy(), pv["s1"].to_numpy()]
    sx = np.r_[pj["sx"].to_numpy(), pv["sx"].to_numpy()]
    y = np.r_[pj["label"].to_numpy(), pv["label"].to_numpy()].astype(np.float32)
    del pj, pv
    fold = s1 % 2
    p1 = np.empty(len(y), dtype=np.float32)
    for f in (0, 1):
        idx = np.flatnonzero(fold == f)
        m = xgb.train(judge.PARAMS, xgb.QuantileDMatrix(X[idx], y[idx]), args.rounds_s1)
        m.save_model(work("models", f"judge_f{f}.json"))
        p1[fold != f] = judge.predict(m, X[fold != f])
        log(f"final stage-1 fold {f}: {args.rounds_s1} rounds on {len(idx)} pairs")
        del m
        gc.collect()
    S2 = stage2_features(tok, s1, sx, p1)
    m2 = xgb.train(judge.PARAMS, xgb.QuantileDMatrix(np.hstack([X, S2]), y), args.rounds_s2)
    m2.save_model(work("models", "judge_s2.json"))
    log(f"final stage-2: {args.rounds_s2} rounds on {len(y)} pairs")


def stage_stage3_features(args):
    """Stage-3 generator-structure features for V (--split train) or test, from the stage-2 probabilities."""
    from .stage3 import split_features
    if args.split == "train":
        pr = pl.read_parquet(work("feat", "train_V_pairs.parquet"), columns=["s1", "sx"])
        p, prep, out = np.load(work("pred", "train_V_p2.npy")).astype(np.float32), "train", work("feat", "stage3_V.parquet")
    else:
        pr = pl.read_parquet(work("pred", "test_pairs_p2.parquet"))
        p, prep, out = pr["p"].to_numpy().astype(np.float32), "test", work("feat", "stage3_test.parquet")
    split_features(prep, pr["s1"].to_numpy(), pr["sx"].to_numpy(), p, out, log=log)
    log("stage-3 features", out)


def stage_stage3_train(args):
    """Train the stage-3 recalibrator on V (labels) and save models/stage3.txt."""
    from .stage3 import NEW_F, train
    pv = pl.read_parquet(work("feat", "train_V_pairs.parquet"))
    f = pl.read_parquet(work("feat", "stage3_V.parquet")).sort("i")
    assert f.height == pv.height
    feats = {c: f[c].to_numpy().astype(np.float32) for c in NEW_F}
    m = train(np.load(work("feat", "train_V.npy"), mmap_mode="r"), feats, np.load(work("pred", "train_V_p2.npy")),
              pv["label"].to_numpy())
    m.save_model(work("models", "stage3.txt"))
    log("stage-3 model saved")


def stage_stage3_predict(args):
    """Stage-3 probabilities for test (pred/test_pairs_p3.parquet) with the shift and unseen-country guards."""
    import lightgbm as lgb

    from .features import FEATURES
    from .stage3 import guard, predict
    pr = pl.read_parquet(work("pred", "test_pairs_p2.parquet"))
    p2 = pr["p"].to_numpy().astype(np.float32)
    X = np.load(work("feat", "test.npy"), mmap_mode="r")
    assert X.shape[0] == pr.height
    p3 = predict(lgb.Booster(model_file=work("models", "stage3.txt")), X, work("feat", "stage3_test.parquet"), p2)
    seen = set(load_prep("train", "s1", ["country"])["country"].unique().to_list())
    unseen = ~np.isin(load_prep("test", "s1", ["country"])["country"].to_numpy()[pr["s1"].to_numpy()], sorted(seen))
    p = guard(p2, p3, np.asarray(X[:, FEATURES.index("hn_in_set")]) > 0.5, unseen)
    pr.with_columns(pl.Series("p", p)).write_parquet(work("pred", "test_pairs_p3.parquet"))
    log(f"stage-3 applied to {int((~unseen).sum())} pairs of seen countries; mean |p3-p2| {np.abs(p - p2)[~unseen].mean():.4f}")


def stage_predict_stage2(args):
    """Test predictions with the stage-2 judge from saved test features (written by predict-test)."""
    import json

    import xgboost as xgb

    from . import judge
    from .context import stage2_features
    from .features import load_token_arrays
    from .io_utils import write_submission
    with open(work("models", "decision_s2.json")) as f:
        decision = json.load(f)
    decision["country_thr"] = _country_map(args.country_thr)
    decision["country_shift"] = _country_map(args.country_shift)
    tok = _load_tok("test")
    scored = pl.read_parquet(work("pred", "test_pairs_p.parquet"))
    s1r, sxr = scored["s1"].to_numpy(), scored["sx"].to_numpy()
    X = np.load(work("feat", "test.npy"), mmap_mode="r")
    fold_models = [xgb.Booster(model_file=work("models", f"judge_f{f}.json")) for f in (0, 1)]
    m2 = xgb.Booster(model_file=work("models", "judge_s2.json"))
    p1 = np.mean([judge.predict(m, X) for m in fold_models], axis=0)
    S2 = stage2_features(tok, s1r, sxr, p1)
    p2 = np.empty(len(p1), dtype=np.float32)
    for b in range(0, len(p1), FEAT_CHUNK):
        p2[b:b + FEAT_CHUNK] = judge.predict(m2, np.hstack([np.asarray(X[b:b + FEAT_CHUNK], dtype=np.float32),
                                                            S2[b:b + FEAT_CHUNK]]))
    pl.DataFrame({"s1": s1r, "sx": sxr, "p": p2}).write_parquet(work("pred", "test_pairs_p2.parquet"))
    s1 = load_prep("test", "s1", ["entity_id", "country"])
    sx_ids = load_prep("test", "sx", ["entity_id"])["entity_id"].to_numpy()
    reject = shift_rule_mask(X) if args.shift_rule else None
    mask = _decide(s1r, sxr, p2, decision, s1_country=s1["country"].to_numpy(), reject=reject)
    write_submission(config.OUT_DIR, s1["entity_id"].to_list(), sx_ids, s1r, sxr, mask)
    log("stage-2 matched pairs", int(mask.sum()), decision, "shift rule" if args.shift_rule else "")
    del sx_ids, X
    gc.collect()
    _run_validator()


SHIFT_RULE_COLS = ("hn_in_set", "hn_off", "hn_sib_1")


def shift_rule_mask(X):
    """Pairs whose SX house number is the S1 number plus a distractor shift of 3 or more although other
    name-similar copies confirm the S1 number. Train has such true copies only rarely (0.002 predicted per S1
    on V, cost of dropping them 0.00013 V), while test US predicts 0.11 per S1 of them, mostly fake businesses
    that come with 2+ copies at the shifted number. X: a feature matrix (70-column files predate 'acro')."""
    from .features import FEATURES
    names = FEATURES if X.shape[1] == len(FEATURES) else [f for f in FEATURES if f != "acro"]
    M = np.asarray(X[:, [names.index(c) for c in SHIFT_RULE_COLS]], dtype=np.float32)
    return (M[:, 0] > 0.5) & (M[:, 1] >= 3) & (M[:, 2] >= 1)


SOURCE_CAPS = (5, 6)   # train truth: at most 5 copies from S2 and 6 from S3 per S1, no exception in 2.2M S1


def cap_per_source(s1r, is_s3, p, mask, caps=SOURCE_CAPS):
    """Keep at most caps[0] S2 and caps[1] S3 matches per S1 (highest p first)."""
    idx = np.flatnonzero(mask)
    src = is_s3[idx].astype(np.int64)
    order = np.lexsort((-p[idx], src, s1r[idx]))
    g = s1r[idx][order] * 2 + src[order]
    start = np.r_[True, g[1:] != g[:-1]]
    rank = np.arange(len(g)) - np.maximum.accumulate(np.where(start, np.arange(len(g)), 0))
    limit = np.where(src[order] == 1, caps[1], caps[0])
    out = mask.copy()
    out[idx[order][rank >= limit]] = False
    return out


def _lookalike_post(split, s1r, sxr, X, rules=None):
    """Returns f(p, keep) -> reject mask of the lookalike rules (src/lookalike.py) for these pairs."""
    from .features import FEATURES
    from .lookalike import NEED_COLS, extra_legal, extra_words, legal_dropped, lookalike_reject
    names = FEATURES if X.shape[1] == len(FEATURES) else [f for f in FEATURES if f != "acro"]
    M = np.asarray(X[:, [names.index(c) for c in NEED_COLS]], dtype=np.float32)
    cols = {c: M[:, i] for i, c in enumerate(NEED_COLS)}
    s1 = load_prep(split, "s1", ["legal", "country"])
    sx_legal = load_prep(split, "sx", ["legal"])["legal"].to_list()
    xleg = extra_legal(s1["legal"].to_list(), sx_legal, s1r, sxr)
    ldrop = legal_dropped(s1["legal"].to_list(), sx_legal, s1r, sxr)
    del sx_legal
    tr1 = load_prep("train", "s1", ["country", "legal"])
    seen = set(tr1["country"].unique().to_list())
    unseen = ~np.isin(s1["country"].to_numpy()[s1r], sorted(seen))
    # countries whose train true copies (almost) never swap legal forms get rule RL
    from .lookalike import LEGAL_SWAP_STRICT, legal_swapped
    gt = pl.read_parquet(work("prep", "train_gt_rows.parquet"))
    g1, gx = gt["s1_row"].to_numpy(), gt["sx_row"].to_numpy()
    sw = legal_swapped(tr1["legal"].to_list(), load_prep("train", "sx", ["legal"])["legal"].to_list(), g1, gx)
    tc = tr1["country"].to_numpy()[g1]
    strict_c = [c for c in sorted(seen) if sw[tc == c].mean() < LEGAL_SWAP_STRICT]
    log("legal-swap rate of train true copies:", {c: round(float(sw[tc == c].mean()), 4) for c in sorted(seen)},
        "-> RL for", strict_c)
    strict = np.isin(s1["country"].to_numpy()[s1r], strict_c)
    del tr1, gt, g1, gx, sw, tc
    with np.load(work("tok", f"{split}.npz")) as z:
        tok = {k: z[k] for k in ("ns1_ptr", "ns1_ids", "nsx_ptr", "nsx_ids")}
    vocab = pl.read_parquet(work("tok", f"{split}_vocab_n.parquet"))["token"].to_list()
    xword = extra_words(tok, vocab, s1r, sxr, s1["country"].to_numpy()[s1r])
    from .lookalike import strong_fake_words
    xstrong = strong_fake_words(tok, vocab, s1r, sxr)
    rw = None
    if rules and "RW" in rules:
        from .lookalike import common_word_swap
        rw = common_word_swap(tok, vocab, s1r, sxr, unseen & (cols["hn_off"] == 0) & (cols["addr_empty_x"] < 0.5))
        log("RW candidates (unseen country, same number, one common word swapped):", int(rw.sum()))
    del tok, vocab

    def post(p, keep):
        kw = {"rules": tuple(rules)} if rules else {}
        rej, masks = lookalike_reject(s1r, p, keep, cols, xleg, xword, ldrop=ldrop, unseen=unseen, strict=strict,
                                      rw=rw, xstrong=xstrong, **kw)
        log("lookalike rules removed", {k: int(v.sum()) for k, v in masks.items()}, "total", int(rej.sum()))
        return rej
    return post


def _word_boost(split, s1r, sxr, X):
    """lookalike.word_boost for these pairs (countries unseen in training only)."""
    from .features import FEATURES
    from .lookalike import BOOST_COLS, word_boost
    names = FEATURES if X.shape[1] == len(FEATURES) else [f for f in FEATURES if f != "acro"]
    M = np.asarray(X[:, [names.index(c) for c in BOOST_COLS]], dtype=np.float32)
    cols = {c: M[:, i] for i, c in enumerate(BOOST_COLS)}
    seen = set(load_prep("train", "s1", ["country"])["country"].unique().to_list())
    unseen = ~np.isin(load_prep(split, "s1", ["country"])["country"].to_numpy()[s1r], sorted(seen))
    with np.load(work("tok", f"{split}.npz")) as z:
        tok = {k: z[k] for k in ("ns1_ptr", "ns1_ids", "nsx_ptr", "nsx_ids")}
    vocab = pl.read_parquet(work("tok", f"{split}_vocab_n.parquet"))["token"].to_list()
    return word_boost(tok, vocab, s1r, sxr, cols, unseen)


def _decide(s1r, sxr, p, decision, s1_country=None, reject=None, post=None, is_s3=None, boost=None):
    """Assignment, then the decision rule. decision["country_thr"] (optional, threshold rule only)
    overrides the threshold for S1s of the named countries (s1_country: per-S1-row country array).
    decision["country_shift"] (optional, any rule) adds a logit shift to the named countries' p first.
    reject (optional bool per pair): pairs that are never matched (p set to 0 before assignment).
    post (optional f(p, keep) -> bool per pair): pairs dropped after assignment, before the cut.
    is_s3 (optional bool per pair): enables the per-source caps after the cut."""
    from .decide import assign_best, expected_f05_cut, threshold_cut
    if boost is not None:
        from .lookalike import BOOST_P
        p = np.where(boost, np.maximum(np.asarray(p, dtype=np.float32), np.float32(BOOST_P)), p)
        log(f"boosted {int(boost.sum())} pairs by rule")
    if reject is not None:
        p = np.where(reject, np.float32(0), np.asarray(p, dtype=np.float32))
        log(f"rejected {int(reject.sum())} pairs by rule")
    shifts = decision.get("country_shift") or {}
    if shifts:
        p = np.asarray(p, dtype=np.float32).copy()
        for c, d in shifts.items():
            m = s1_country[s1r] == c
            z = np.log(np.clip(p[m], 1e-6, 1 - 1e-6) / np.clip(1 - p[m], 1e-6, 1)) + d
            p[m] = 1.0 / (1.0 + np.exp(-z))
    keep = assign_best(sxr, p)
    if post is not None:
        keep &= ~post(p, keep)
    mask = np.zeros(len(p), dtype=bool)
    idx = np.flatnonzero(keep)
    if decision["rule"] == "expected_f":
        sub = expected_f05_cut(s1r[idx], p[idx])
    else:
        thr = np.full(len(idx), decision["threshold"], dtype=np.float32)
        for c, t in (decision.get("country_thr") or {}).items():
            thr[s1_country[s1r[idx]] == c] = t
        sub = threshold_cut(p[idx], thr)
    mask[idx[sub]] = True
    if is_s3 is not None:
        capped = cap_per_source(s1r, is_s3, p, mask)
        log("per-source caps removed", int(mask.sum() - capped.sum()))
        mask = capped
    return mask


def _run_validator():
    """Validate the scored file only. The candidate file is skipped on purpose: the validator keeps every
    candidate id in Python sets (~7 GB at 69M ids), and matches are candidate rows by construction.
    cwd=OUT_DIR so the validator's default 'output/candidate_pairs.tsv' is not found and gets skipped."""
    import subprocess
    import sys
    validator = os.path.join(config.ROOT, "student_resource", "utils", "validate_submission.py")
    if not os.path.isfile(validator):
        log("validator not found at", validator, "- skipping format check")
        return
    subprocess.run([sys.executable, validator, "--matching", os.path.join(config.OUT_DIR, "matching_results.tsv"),
                    "--test-dir", os.path.join(config.DATA_DIR, "test")], cwd=config.OUT_DIR, check=True)


def stage_predict_test(args):
    import json

    import xgboost as xgb

    from . import judge
    from .features import PairFeaturizer, add_list_features, load_token_arrays
    from .io_utils import write_submission
    with open(work("models", "decision.json")) as f:
        decision = json.load(f)
    with open(work("models", "stage_a.json")) as f:
        t_a = json.load(f)["threshold"]
    country = load_prep("test", "s1", ["country"])["country"].to_numpy()
    names, code = np.unique(country, return_inverse=True)
    code = code.astype(np.int8)
    tok = _load_tok("test")
    models_a = [xgb.Booster(model_file=work("models", f"stage_a_{f}.json")) for f in (0, 1)]

    # list + competition features and the stage-A filter, one country at a time. Each country's candidates
    # are read lazily: holding the whole 79M-row file pushed the process past 16 GB private memory.
    kept, n_before = [], 0
    for k, c in enumerate(names):
        rows = pl.Series(np.flatnonzero(code == k).astype(np.int32))
        cc = pl.scan_parquet(work("cand", "test.parquet")).filter(pl.col("s1").is_in(rows.implode())).collect()
        n_before += cc.height
        cc = _with_addr_empty(_with_competition(add_list_features(cc)), tok)
        pa = _stage_a_predict(cc, models_a)
        cc = cc.with_columns(pl.Series("pA", pa)).filter(pl.col("pA") >= t_a)
        log(f"  {c}: kept {cc.height} of {len(pa)} candidate pairs")
        kept.append(cc)
        del pa, rows
        gc.collect()
    cand = pl.concat(kept).sort("s1", "rank")
    del kept
    gc.collect()
    log("stage A kept", cand.height, "of", n_before, f"({cand.height / len(country):.1f}/S1)")

    # word log-odds learned on train, carried to test token ids through the vocabularies
    lo = np.load(work("models", "word_lo_train.npz"))
    vtr = pl.read_parquet(work("tok", "train_vocab_n.parquet"))["token"].to_list()
    vte = pl.read_parquet(work("tok", "test_vocab_n.parquet"))["token"].to_list()
    from .context import WORD_EQUIV
    lo_test = []
    for arr in (lo["extra"], lo["miss"]):
        m = dict(zip(vtr, arr.tolist()))
        lo_test.append(np.array([m.get(t, m.get(WORD_EQUIV.get(t, t), 0.0)) for t in vte], dtype=np.float32))
    cand = _add_context(cand, tok, *lo_test)
    del lo, vtr, vte, lo_test
    gc.collect()

    from .features import FEATURES
    s1, sx = _load_split_frames("test")   # string columns: only the featurizer needs them
    fz = PairFeaturizer(s1, sx, tok)
    booster = xgb.Booster(model_file=work("models", "judge_v1.json"))
    p = np.empty(cand.height, dtype=np.float32)
    xout = np.lib.format.open_memmap(work("feat", "test.npy"), mode="w+", dtype=np.float16,
                                     shape=(cand.height, len(FEATURES)))  # kept for the stage-2 judge
    for b in range(0, cand.height, FEAT_CHUNK):
        Xc = fz.compute(cand.slice(b, FEAT_CHUNK))
        xout[b:b + FEAT_CHUNK] = Xc
        p[b:b + FEAT_CHUNK] = judge.predict(booster, Xc)
        log(f"  test scored {min(b + FEAT_CHUNK, cand.height)}/{cand.height}")
    xout.flush()
    del xout
    s1r, sxr = cand["s1"].to_numpy().copy(), cand["sx"].to_numpy().copy()
    pl.DataFrame({"s1": s1r, "sx": sxr, "p": p}).write_parquet(work("pred", "test_pairs_p.parquet"))
    s1_ids, sx_ids = s1["entity_id"].to_list(), sx["entity_id"].to_numpy()
    del cand, fz, s1, sx, tok
    gc.collect()
    reject = shift_rule_mask(np.load(work("feat", "test.npy"), mmap_mode="r")) if args.shift_rule else None
    mask = _decide(s1r, sxr, p, decision, s1_country=country, reject=reject)
    write_submission(config.OUT_DIR, s1_ids, sx_ids, s1r, sxr, mask)
    log("matched pairs", int(mask.sum()), "S1 with >=1 match", len(np.unique(s1r[mask])), "of", len(s1_ids))
    del sx_ids, s1r, sxr, p, mask
    gc.collect()
    _run_validator()


def _style_additions(s1r, sxr, p, X, mask0, keep):
    """lookalike-independent additions for countries unseen in training (src/style.py); bool per pair."""
    from .features import FEATURES
    from .io_utils import read_source
    from .style import ALL_CATS, address_style, common_tokens, name_category, street_mismatch, style_additions
    names = FEATURES if X.shape[1] == len(FEATURES) else [f for f in FEATURES if f != "acro"]
    col = lambda c: np.asarray(X[:, names.index(c)], dtype=np.float32)
    s1 = load_prep("test", "s1", ["country", "name_core", "addr_norm"])
    seen = set(load_prep("train", "s1", ["country"])["country"].unique().to_list())
    unseen_c = sorted(set(s1["country"].unique().to_list()) - seen)
    if not unseen_c:
        return np.zeros(len(p), dtype=bool)
    country = s1["country"].to_numpy()
    unseen = np.isin(country[s1r], unseen_c)
    sx = load_prep("test", "sx", ["entity_id", "country", "name_core", "addr_norm", "src"])
    # raw address style of the SX records of the unseen countries
    style = {}
    for k in (2, 3):
        raw = read_source("test", k).filter(pl.col("country").is_in(unseen_c))
        for e, a, c in zip(raw["entity_id"].to_list(), raw["business_address"].to_list(), raw["country"].to_list()):
            style[e] = address_style(a, c)
        del raw
    ids = sx["entity_id"].to_list()
    nf_sx = np.array([style.get(e, (-1, -1))[0] for e in ids], dtype=np.int16)
    reg_sx = np.array([style.get(e, (-1, -1))[1] for e in ids], dtype=np.int8)
    del style, ids
    matched = np.unique(sxr[mask0])
    cand = unseen & (col("hn_off") == 0) & (col("addr_empty_x") < 0.5) & ~mask0 & ~np.isin(sxr, matched)
    rows = np.flatnonzero(cand | (keep & (p >= 0.99) & unseen & (col("hn_off") == 0)))   # candidates + anchors
    ncat = np.full(len(p), "", dtype=object)
    n1 = s1["name_core"].to_numpy()
    nx = sx["name_core"].to_numpy()
    ncat[rows] = [name_category(n1[a], nx[b]) for a, b in zip(s1r[rows], sxr[rows])]
    cand &= np.isin(ncat, ALL_CATS)
    common = common_tokens(s1.filter(pl.col("country").is_in(unseen_c))["addr_norm"].to_list())
    a1, ax = s1["addr_norm"].to_numpy(), sx["addr_norm"].to_numpy()
    smis = np.zeros(len(p), dtype=np.int8)
    cr = np.flatnonzero(cand)
    smis[cr] = [street_mismatch(a1[a], ax[b], common) for a, b in zip(s1r[cr], sxr[cr])]
    add = style_additions(s1r, sxr, p, keep, mask0, cand, nf_sx[sxr], reg_sx[sxr], sx["src"].to_numpy()[sxr], ncat, smis)
    log(f"style additions: {int(cand.sum())} eligible, {int(add.sum())} added")
    return add


def _country_map(spec):
    return {k: float(v) for k, v in (kv.split("=") for kv in spec.split(","))} if spec else {}


def stage_rethreshold(args):
    """Re-decide saved test probabilities (stage 1 by default, --stage2 for the stage-2 ones)."""
    import json

    from .io_utils import write_submission
    with open(work("models", "decision_s2.json" if (args.stage2 or args.stage3) else "decision.json")) as f:
        decision = json.load(f)
    decision.update({"rule": args.rule, "threshold": args.thr if args.thr is not None else decision["threshold"]})
    decision["country_thr"] = _country_map(args.country_thr)
    decision["country_shift"] = _country_map(args.country_shift)
    s1 = load_prep("test", "s1", ["entity_id", "country"])
    sx_ids = load_prep("test", "sx", ["entity_id"])["entity_id"].to_numpy()
    scored = pl.read_parquet(work("pred", "test_pairs_p3.parquet" if args.stage3 else
                                  "test_pairs_p2.parquet" if args.stage2 else "test_pairs_p.parquet"))
    s1r, sxr, p = scored["s1"].to_numpy(), scored["sx"].to_numpy(), scored["p"].to_numpy()
    del scored
    reject, post, is_s3, boost = None, None, None, None
    if args.shift_rule or args.lookalike or args.caps or args.word_boost:  # test.npy rows are aligned with pairs
        X = np.load(work("feat", "test.npy"), mmap_mode="r")
        assert X.shape[0] == len(p), "feat/test.npy is not from the run that scored these pairs"
        if args.shift_rule:
            reject = shift_rule_mask(X)
        if args.lookalike:
            post = _lookalike_post("test", s1r, sxr, X, args.rules.split(",") if args.rules else None)
        if args.word_boost:
            boost = _word_boost("test", s1r, sxr, X)
        if args.caps:
            from .features import FEATURES
            names = FEATURES if X.shape[1] == len(FEATURES) else [f for f in FEATURES if f != "acro"]
            is_s3 = np.asarray(X[:, names.index("is_s3")]) > 0.5
        del X
    mask = _decide(s1r, sxr, p, decision, s1_country=s1["country"].to_numpy(), reject=reject, post=post,
                   is_s3=is_s3, boost=boost)
    if args.style_add:
        from .decide import assign_best
        X = np.load(work("feat", "test.npy"), mmap_mode="r")
        add = _style_additions(s1r, sxr, p, X, mask, assign_best(sxr, p))
        del X
        p = np.where(add, np.float32(0.95) + np.float32(0.05) * p, p)   # order among additions preserved
        mask = _decide(s1r, sxr, p, decision, s1_country=s1["country"].to_numpy(), reject=reject, post=post,
                       is_s3=is_s3, boost=boost)
    write_submission(config.OUT_DIR, s1["entity_id"].to_list(), sx_ids, s1r, sxr, mask)
    log("rethreshold", decision, "matched pairs", int(mask.sum()))
    del sx_ids, s1r, sxr, p, mask
    gc.collect()
    _run_validator()


STAGES = {"prepare": stage_prepare, "train-encoder": stage_train_encoder, "encode": stage_encode,
          "candidates": stage_candidates, "address-pass": stage_address_pass, "tokens": stage_tokens,
          "vocab": stage_vocab, "stage2": stage_stage2, "predict-stage2": stage_predict_stage2,
          "stage2-final": stage_stage2_final, "stage3-features": stage_stage3_features,
          "stage3-train": stage_stage3_train, "stage3-predict": stage_stage3_predict,
          "features": stage_features, "train-judge": stage_train_judge,
          "validate": stage_validate, "predict-test": stage_predict_test, "rethreshold": stage_rethreshold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--k-addr", type=int, default=0, help="extra neighbours by address cosine (candidates)")
    ap.add_argument("--queries", default="auto", choices=["auto", "E"],
                    help="train candidates: auto = J+V S1s, E = encoder-split S1s (competitor lists)")
    ap.add_argument("--rule", default="threshold", choices=["threshold", "expected_f"])
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--country-thr", default="", help="per-country thresholds, e.g. France=0.85,India=0.7")
    ap.add_argument("--country-shift", default="", help="per-country logit shifts of p, e.g. France=-0.9")
    ap.add_argument("--stage2", action="store_true", help="rethreshold: use the stage-2 probabilities")
    ap.add_argument("--stage3", action="store_true", help="rethreshold: use the stage-3 probabilities")
    ap.add_argument("--shift-rule", action="store_true",
                    help="never match an SX shifted by a distractor offset >= 3 when copies confirm the S1 number")
    ap.add_argument("--lookalike", action="store_true", help="rethreshold: drop lookalike fake groups (src/lookalike.py)")
    ap.add_argument("--caps", action="store_true", help="rethreshold: at most 5 S2 and 6 S3 matches per S1")
    ap.add_argument("--rules", default="", help="lookalike rules to apply (default src/lookalike.RULES)")
    ap.add_argument("--style-add", action="store_true",
                    help="rethreshold: add unmatched same-number copies in unseen countries whose address style matches the S1's copies")
    ap.add_argument("--word-boost", action="store_true",
                    help="rethreshold: raise exact-address dual-role word copies in unseen countries (lookalike.word_boost)")
    ap.add_argument("--rounds-s1", type=int, default=640, help="stage2-final: rounds of each stage-1 fold judge")
    ap.add_argument("--rounds-s2", type=int, default=640, help="stage2-final: rounds of the stage-2 judge")
    ap.add_argument("--drop-frac", type=float, default=0.0,
                    help="features: simulate this share of S1s missing (as in test) for training and V")
    args = ap.parse_args()
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
