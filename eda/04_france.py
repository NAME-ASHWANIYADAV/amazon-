"""Look at test France records: random S1 plus S2/S3 records sharing the rarest S1 name token."""
import re
import sys

import polars as pl

C = r"C:\amazon_ml\cache"
n_show = int(sys.argv[1]) if len(sys.argv) > 1 else 10
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

s1 = pl.read_parquet(f"{C}/test_s1.parquet").filter(pl.col("country") == "France")
sx = pl.concat([pl.read_parquet(f"{C}/test_s2.parquet"), pl.read_parquet(f"{C}/test_s3.parquet")]).filter(
    pl.col("country") == "France")

tok = lambda s: [t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2]
sx = sx.with_columns(pl.col("business_name").str.to_lowercase().alias("nl"),
                     pl.col("business_address").str.to_lowercase().alias("al"))
freq = (sx.select(pl.col("nl").str.extract_all(r"[a-z0-9]+").alias("t")).explode("t")
        .group_by("t").len())
fmap = dict(zip(freq["t"].to_list(), freq["len"].to_list()))

for r in s1.sample(n_show, seed=seed).iter_rows(named=True):
    print(f"\n[S1] {r['entity_id']} | {r['business_name']} | {r['business_address']}")
    toks = tok(r["business_name"])
    if not toks:
        continue
    rare = min(toks, key=lambda t: fmap.get(t, 10 ** 9) if fmap.get(t) else 10 ** 9)
    street = [t for t in tok(r["business_address"]) if not t.isdigit()]
    m = sx.filter(pl.col("nl").str.contains(rf"\b{re.escape(rare)}\b"))
    print(f"     rare token '{rare}' freq={fmap.get(rare)}  hits={m.height}")
    for q in m.head(12).iter_rows(named=True):
        print(f"     {q['entity_id'][:2]} | {q['business_name']} | {q['business_address']}")
