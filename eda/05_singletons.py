"""Do singleton S1 entities have lookalike S2/S3 records? Search by rarest name token."""
import re
import sys

import polars as pl

C = r"C:\amazon_ml\cache"
country = sys.argv[1] if len(sys.argv) > 1 else "US"
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 8

s1 = pl.read_parquet(f"{C}/train_s1.parquet").filter(pl.col("country") == country)
sx = pl.concat([pl.read_parquet(f"{C}/train_s2.parquet"), pl.read_parquet(f"{C}/train_s3.parquet")]).filter(
    pl.col("country") == country).with_columns(pl.col("business_name").str.to_lowercase().alias("nl"))
pairs = pl.read_parquet(f"{C}/train_gt_pairs.parquet")
owner = dict(zip(pairs["sx"].to_list(), pairs["s1"].to_list()))

freq = sx.select(pl.col("nl").str.extract_all(r"[a-z0-9]+").alias("t")).explode("t").group_by("t").len()
fmap = dict(zip(freq["t"].to_list(), freq["len"].to_list()))
single = s1.join(pairs.select(pl.col("s1").alias("entity_id")).unique(), on="entity_id", how="anti")
s1name = dict(zip(s1["entity_id"].to_list(), s1["business_name"].to_list()))

for r in single.sample(n_show, seed=7).iter_rows(named=True):
    print(f"\n[S1 singleton] {r['business_name']} | {r['business_address']}")
    toks = [t for t in re.findall(r"[a-z0-9]+", r["business_name"].lower()) if len(t) > 2 and fmap.get(t)]
    if not toks:
        print("   no shared tokens")
        continue
    rare = min(toks, key=lambda t: fmap[t])
    m = sx.filter(pl.col("nl").str.contains(rf"\b{re.escape(rare)}\b"))
    print(f"   rare token '{rare}' freq={fmap[rare]}")
    for q in m.head(8).iter_rows(named=True):
        o = owner.get(q["entity_id"])
        tag = f"-> belongs to S1 '{s1name.get(o)}'" if o else "-> UNMATCHED distractor"
        print(f"   {q['entity_id'][:2]} | {q['business_name']} | {q['business_address']}  {tag}")
