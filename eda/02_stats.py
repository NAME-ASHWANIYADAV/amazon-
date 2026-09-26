"""Core dataset statistics: countries, GT structure, singletons, 1-to-many constraint, scripts."""
import gc

import polars as pl

C = r"C:\amazon_ml\cache"
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)
pl.Config.set_fmt_str_lengths(120)


def load(split, s):
    return pl.read_parquet(f"{C}/{split}_s{s}.parquet")


print("=== per-file stats", flush=True)
for split in ("train", "test"):
    for s in (1, 2, 3):
        df = load(split, s)
        r = df.group_by("country").agg(
            pl.len().alias("n"),
            (pl.col("business_name").str.strip_chars() == "").sum().alias("empty_name"),
            (pl.col("business_address").str.strip_chars() == "").sum().alias("empty_addr"),
            pl.col("business_address").str.to_lowercase().str.contains(r"\bnull\b").mean().round(4).alias("null_in_addr"),
            (~pl.col("business_name").str.contains(r"^[\x00-\x7F]*$")).mean().round(4).alias("nonascii_name"),
            pl.col("business_name").str.contains("[\u0900-\u097F]").mean().round(4).alias("deva_name"),
            pl.col("business_address").str.contains("[\u0900-\u0DFF]").mean().round(4).alias("indic_addr"),
            pl.col("business_name").str.len_chars().mean().round(1).alias("name_len"),
            pl.col("business_address").str.len_chars().mean().round(1).alias("addr_len"),
        ).sort("country")
        print(split, s)
        print(r, flush=True)
        del df
        gc.collect()

s1 = load("train", 1).select("entity_id", "country")
gt = pl.read_parquet(f"{C}/train_gt.parquet")
pairs = pl.read_parquet(f"{C}/train_gt_pairs.parquet")

print("\n=== GT coverage", flush=True)
print("gt rows", gt.height, "unique s1", gt["source1_entity_id"].n_unique(), "s1 file", s1.height)
print("gt s1 ids all in s1 file:", gt.join(s1, left_on="source1_entity_id", right_on="entity_id", how="anti").height == 0)

n = pairs.group_by("s1").len()
counts = (s1.select(pl.col("entity_id").alias("s1"), "country")
          .join(n, on="s1", how="left").with_columns(pl.col("len").fill_null(0)))
print("\n=== matches per S1 (overall)")
print(counts.group_by("len").len(name="n_s1").sort("len").with_columns(
    (pl.col("n_s1") / counts.height).round(4).alias("frac")), flush=True)
print("\n=== singleton rate & mean matches by country")
print(counts.group_by("country").agg(
    pl.len().alias("n"), (pl.col("len") == 0).mean().round(4).alias("singleton_rate"),
    pl.col("len").mean().round(3).alias("mean_matches"), pl.col("len").max().alias("max")), flush=True)

pairs = pairs.with_columns(pl.col("sx").str.slice(0, 2).alias("src"))
per = pairs.group_by("s1", "src").len().pivot(on="src", index="s1", values="len").fill_null(0)
print("\n=== S2 / S3 matches per non-singleton S1")
print(per.select(pl.col("S2").mean().alias("mean_S2"), pl.col("S3").mean().alias("mean_S3"),
                 (pl.col("S2") == 0).mean().alias("no_S2"), (pl.col("S3") == 0).mean().alias("no_S3")))
print(per.group_by("S2", "S3").len().sort("len", descending=True).head(15), flush=True)

print("\n=== 1-to-many check: does any S2/S3 id appear under >1 S1?")
dup = pairs.group_by("sx").len().filter(pl.col("len") > 1)
print("sx appearing >1 times:", dup.height, "of", pairs["sx"].n_unique(), flush=True)

xs = []
for s in (2, 3):
    x = load("train", s).select(pl.col("entity_id").alias("sx"), pl.col("country").alias("cx"))
    xs.append(x)
    m = pairs.filter(pl.col("src") == f"S{s}").select("sx").unique()
    missing = m.join(x, on="sx", how="anti").height
    print(f"S{s}: records {x.height}, matched to some S1 {m.height} ({m.height / x.height:.4f}), "
          f"gt ids missing from file: {missing}", flush=True)
allx = pl.concat(xs)
del xs

print("\n=== country consistency of matched pairs")
j = pairs.join(s1.rename({"entity_id": "s1", "country": "c1"}), on="s1").join(allx, on="sx")
print(j.group_by("c1", "cx").len().sort("len", descending=True), flush=True)

print("\n=== matched share of S2/S3 by country")
mm = allx.join(pairs.select("sx", "src"), on="sx", how="left").with_columns(pl.col("src").is_not_null().alias("m"))
print(mm.with_columns(pl.col("sx").str.slice(0, 2).alias("source")).group_by("source", "cx").agg(
    pl.col("m").mean().round(4).alias("matched_share"), pl.len()).sort("source", "cx"), flush=True)

print("\n=== test sizes by country (S2+S3 per S1 ratio)")
t1 = load("test", 1).group_by("country").len().rename({"len": "n1"})
t2 = load("test", 2).group_by("country").len().rename({"len": "n2"})
t3 = load("test", 3).group_by("country").len().rename({"len": "n3"})
tr1 = load("train", 1).group_by("country").len().rename({"len": "n1"})
tr2 = load("train", 2).group_by("country").len().rename({"len": "n2"})
tr3 = load("train", 3).group_by("country").len().rename({"len": "n3"})
for name, a, b, c in (("train", tr1, tr2, tr3), ("test", t1, t2, t3)):
    print(name)
    print(a.join(b, on="country", how="full", coalesce=True).join(c, on="country", how="full", coalesce=True)
          .with_columns(((pl.col("n2") + pl.col("n3")) / pl.col("n1")).round(3).alias("ratio")).sort("country"))
