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
    if args.split == "train":
        q_rows = np.flatnonzero((s1["part"] != "E").to_numpy())
    else:
        q_rows = np.arange(s1.height)
    cand = build_candidates(s1["country"].to_numpy(), sx["country"].to_numpy(), q_rows, s1_emb, sx_emb,
                            config.TOP_K, sx_addr_empty=sx["addr_empty"].to_numpy(), k_addr=args.k_addr, log=log)
    cand.write_parquet(work("cand", f"{args.split}.parquet"))
    log("candidates", cand.shape)
    if args.split == "train":
        gt = pl.read_parquet(work("prep", "train_gt_rows.parquet"))
        for p in ("J", "V"):
            log(f"recall on {p}:")
            recall_report(cand, gt, np.flatnonzero((s1["part"] == p).to_numpy()), log=log)


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
    cand = add_address_pass(pl.read_parquet(backup), s1["country"].to_numpy(), sx["country"].to_numpy(),
                            np.load(work("emb", f"{args.split}_s1.npy"), mmap_mode="r"),
                            np.load(work("emb", f"{args.split}_sx.npy"), mmap_mode="r"),
                            args.k_addr, sx_addr_empty=sx["addr_empty"].to_numpy(), log=log)
    cand.write_parquet(path)
    log("candidates with address pass", cand.shape)
    if args.split == "train":
        gt = pl.read_parquet(work("prep", "train_gt_rows.parquet"))
        log("recall on V:")
        recall_report(cand, gt, np.flatnonzero((s1["part"] == "V").to_numpy()), log=log)


FEAT_CHUNK = 2_000_000


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
    return s1, sx, PairFeaturizer(s1, sx, load_token_arrays(work("tok", f"{split}.npz")))


def _pruned_candidates(split, floor):
    """Candidates with cos >= floor, plus per-S1 list features computed on the pruned lists."""
    from .features import add_list_features
    cand = pl.read_parquet(work("cand", f"{split}.parquet"))
    if floor is not None:
        cand = cand.filter(pl.col("cos") >= floor)
    return add_list_features(cand)


def _saved_floor():
    import json
    with open(work("models", "floor.json")) as f:
        return json.load(f)["floor"]


def stage_features(args):
    import json

    from .features import FEATURES
    with open(work("models", "floor.json"), "w") as f:
        json.dump({"floor": args.floor}, f)
    s1, sx, fz = _featurizer("train")
    cand = _pruned_candidates("train", args.floor)
    part = s1["part"].to_numpy()
    gt = pl.read_parquet(work("prep", "train_gt_rows.parquet")).with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    cand = cand.join(gt.rename({"s1_row": "s1", "sx_row": "sx"}), on=["s1", "sx"], how="left").with_columns(
        pl.col("label").fill_null(0))
    log("featurizer ready")
    for p in ("J", "V"):
        c = cand.filter(pl.Series(part[cand["s1"].to_numpy()] == p))
        if p == "J":  # early-stopping slice (S1 row % 10 == 0) goes last
            c = c.with_columns((pl.col("s1") % 10 == 0).alias("ev")).sort("ev", "s1", "rank").drop("ev")
        out = np.lib.format.open_memmap(work("feat", f"train_{p}.npy"), mode="w+", dtype=np.float32,
                                        shape=(c.height, len(FEATURES)))
        for b in range(0, c.height, FEAT_CHUNK):
            out[b:b + FEAT_CHUNK] = fz.compute(c.slice(b, FEAT_CHUNK))
            log(f"  {p} features {min(b + FEAT_CHUNK, c.height)}/{c.height}")
        out.flush()
        del out
        c.select("s1", "sx", "label").write_parquet(work("feat", f"train_{p}_pairs.parquet"))
        log(p, "pairs", c.height, "positives", int(c["label"].sum()))


def stage_train_judge(args):
    from . import judge
    X = np.load(work("feat", "train_J.npy"))
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
    import json

    import xgboost as xgb

    from . import judge
    from .decide import assign_best, expected_f05_cut, threshold_cut
    booster = xgb.Booster(model_file=work("models", "judge_v1.json"))
    s1 = load_prep("train", "s1", ["entity_id", "country", "part"])
    sx_ids = load_prep("train", "sx", ["entity_id"])["entity_id"].to_numpy()
    s1_ids = s1["entity_id"].to_numpy()
    country = s1["country"].to_numpy()
    pairs = pl.read_parquet(work("feat", "train_V_pairs.parquet"))
    p = judge.predict(booster, np.load(work("feat", "train_V.npy"), mmap_mode="r"))
    np.save(work("pred", "train_V_p.npy"), p)
    s1r, sxr = pairs["s1"].to_numpy(), pairs["sx"].to_numpy()
    rng = np.random.default_rng(0)
    v_rows = np.flatnonzero((s1["part"] == "V").to_numpy())
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
                "floor": _saved_floor()}
    with open(work("models", "decision.json"), "w") as f:
        json.dump(decision, f)
    log("decision", decision)


def _decide(s1r, sxr, p, decision):
    from .decide import assign_best, expected_f05_cut, threshold_cut
    keep = assign_best(sxr, p)
    mask = np.zeros(len(p), dtype=bool)
    idx = np.flatnonzero(keep)
    sub = (expected_f05_cut(s1r[idx], p[idx]) if decision["rule"] == "expected_f"
           else threshold_cut(p[idx], decision["threshold"]))
    mask[idx[sub]] = True
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
    from .io_utils import write_submission
    with open(work("models", "decision.json")) as f:
        decision = json.load(f)
    s1, sx, fz = _featurizer("test")
    cand = _pruned_candidates("test", decision["floor"])
    booster = xgb.Booster(model_file=work("models", "judge_v1.json"))
    p = np.empty(cand.height, dtype=np.float32)
    for b in range(0, cand.height, FEAT_CHUNK):
        p[b:b + FEAT_CHUNK] = judge.predict(booster, fz.compute(cand.slice(b, FEAT_CHUNK)))
        log(f"  test scored {min(b + FEAT_CHUNK, cand.height)}/{cand.height}")
    np.save(work("pred", "test_p.npy"), p)
    s1r, sxr = cand["s1"].to_numpy().copy(), cand["sx"].to_numpy().copy()
    s1_ids, sx_ids = s1["entity_id"].to_list(), sx["entity_id"].to_numpy()
    del cand, fz, s1, sx
    gc.collect()
    mask = _decide(s1r, sxr, p, decision)
    write_submission(config.OUT_DIR, s1_ids, sx_ids, s1r, sxr, mask)
    log("matched pairs", int(mask.sum()), "S1 with >=1 match", len(np.unique(s1r[mask])), "of", len(s1_ids))
    del sx_ids, s1r, sxr, p, mask
    gc.collect()
    _run_validator()


def stage_rethreshold(args):
    import json

    from .io_utils import write_submission
    with open(work("models", "decision.json")) as f:
        decision = json.load(f)
    decision.update({"rule": args.rule, "threshold": args.thr if args.thr is not None else decision["threshold"]})
    s1 = load_prep("test", "s1", ["entity_id"])
    sx_ids = load_prep("test", "sx", ["entity_id"])["entity_id"].to_numpy()
    cand = _pruned_candidates("test", decision["floor"])
    p = np.load(work("pred", "test_p.npy"))
    s1r, sxr = cand["s1"].to_numpy(), cand["sx"].to_numpy()
    del cand
    mask = _decide(s1r, sxr, p, decision)
    write_submission(config.OUT_DIR, s1["entity_id"].to_list(), sx_ids, s1r, sxr, mask)
    log("rethreshold", decision, "matched pairs", int(mask.sum()))
    del sx_ids, s1r, sxr, p, mask
    gc.collect()
    _run_validator()


STAGES = {"prepare": stage_prepare, "train-encoder": stage_train_encoder, "encode": stage_encode,
          "candidates": stage_candidates, "address-pass": stage_address_pass, "tokens": stage_tokens,
          "features": stage_features, "train-judge": stage_train_judge,
          "validate": stage_validate, "predict-test": stage_predict_test, "rethreshold": stage_rethreshold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--k-addr", type=int, default=0, help="extra neighbours by address cosine (candidates)")
    ap.add_argument("--rule", default="threshold", choices=["threshold", "expected_f"])
    ap.add_argument("--thr", type=float, default=None)
    args = ap.parse_args()
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
