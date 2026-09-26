"""Print example clusters (S1 + its matched S2/S3 records) and unmatched distractors."""
import sys

import polars as pl

C = r"C:\amazon_ml\cache"
country = sys.argv[1] if len(sys.argv) > 1 else "US"
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 12
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

s1 = pl.read_parquet(f"{C}/train_s1.parquet").filter(pl.col("country") == country)
sx = pl.concat([pl.read_parquet(f"{C}/train_s2.parquet"), pl.read_parquet(f"{C}/train_s3.parquet")]).filter(
    pl.col("country") == country)
pairs = pl.read_parquet(f"{C}/train_gt_pairs.parquet")

pick = s1.sample(n_show, seed=seed)
for r in pick.iter_rows(named=True):
    print(f"\n[S1] {r['entity_id']} | {r['business_name']} | {r['business_address']}")
    m = pairs.filter(pl.col("s1") == r["entity_id"]).join(sx, left_on="sx", right_on="entity_id")
    if m.height == 0:
        print("     (singleton)")
    for q in m.sort("sx").iter_rows(named=True):
        print(f"     {q['sx'][:2]} | {q['business_name']} | {q['business_address']}")

print("\n\n===== UNMATCHED S2/S3 records (distractors) =====")
um = sx.join(pairs.select(pl.col("sx").alias("entity_id")), on="entity_id", how="anti").sample(n_show * 2, seed=seed)
for q in um.iter_rows(named=True):
    print(f"  {q['entity_id'][:2]} | {q['business_name']} | {q['business_address']}")
