"""Convert raw TSVs to parquet (quoting disabled, all columns as strings) for fast EDA."""
import os
import time

import polars as pl

DATA = r"C:\amazon_ml\dataset"
CACHE = r"C:\amazon_ml\cache"
os.makedirs(CACHE, exist_ok=True)


def read_tsv(path):
    return pl.read_csv(
        path, separator="\t", quote_char=None, infer_schema=False,
        has_header=True, encoding="utf8", missing_utf8_is_empty_string=True,
    )


for split in ("train", "test"):
    for src in (1, 2, 3):
        t = time.time()
        path = os.path.join(DATA, split, f"{split}_source{src}.tsv")
        df = read_tsv(path)
        df.write_parquet(os.path.join(CACHE, f"{split}_s{src}.parquet"))
        print(split, src, df.shape, df.columns, f"{time.time() - t:.1f}s", flush=True)
        del df

t = time.time()
gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
gt.write_parquet(os.path.join(CACHE, "train_gt.parquet"))
pairs = (
    gt.with_columns(pl.col("matched_entity_ids").str.split(","))
    .explode("matched_entity_ids")
    .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
    .rename({"source1_entity_id": "s1", "matched_entity_ids": "sx"})
)
pairs.write_parquet(os.path.join(CACHE, "train_gt_pairs.parquet"))
print("gt", gt.shape, "pairs", pairs.shape, f"{time.time() - t:.1f}s")
