"""Reading the challenge TSVs and writing submission files."""
import os

import numpy as np
import polars as pl

from .config import DATA_DIR


def read_tsv(path):
    """Challenge files contain stray quote characters: disable quoting and read every column as text."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       has_header=True, missing_utf8_is_empty_string=True)


def read_source(split, src):
    return read_tsv(os.path.join(DATA_DIR, split, f"{split}_source{src}.tsv"))


def read_gt_pairs():
    """Ground truth as a long table of (s1, sx) entity ids."""
    gt = read_tsv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"))
    return (gt.rename({"source1_entity_id": "s1", "matched_entity_ids": "sx"})
            .filter(pl.col("sx") != "")
            .with_columns(pl.col("sx").str.split(","))
            .explode("sx"))


def write_submission(out_dir, s1_ids, sx_ids, s1r, sxr, mask):
    """Stream candidate_pairs.tsv and matching_results.tsv (validator format, one row per S1).

    s1_ids: list of all S1 entity ids (row order); sx_ids: numpy object array of SX ids;
    s1r/sxr: candidate pairs as row indices; mask: True where the pair is a predicted match."""
    os.makedirs(out_dir, exist_ok=True)
    order = np.argsort(s1r, kind="stable")
    s1s, sxs, ms = s1r[order], sxr[order], mask[order]
    bounds = np.searchsorted(s1s, np.arange(len(s1_ids) + 1))
    with open(os.path.join(out_dir, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="\n") as fc, \
            open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for i, s1 in enumerate(s1_ids):
            lo, hi = bounds[i], bounds[i + 1]
            names = sx_ids[sxs[lo:hi]]  # per-S1 slice: never materialise all ~69M names at once
            fc.write(f"{s1}\t{','.join(names)}\n")
            fm.write(f"{s1}\t{','.join(names[ms[lo:hi]])}\n")
