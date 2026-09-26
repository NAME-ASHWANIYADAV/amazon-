"""CLI for every pipeline stage: python -m src.run_pipeline <stage> [options]."""
import argparse
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


STAGES = {"prepare": stage_prepare}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--floor", type=float, default=None)
    args = ap.parse_args()
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
