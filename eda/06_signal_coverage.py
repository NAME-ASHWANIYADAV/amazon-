"""Design evidence: how often true pairs vs. lookalike negatives share name tokens, house numbers,
address words. Negatives = non-matching S2/S3 records that share the S1's rarest name token."""
import re
import unicodedata

import polars as pl

C = r"C:\amazon_ml\cache"
N_S1 = 60_000  # sampled S1 entities per country
LEGAL = set("""inc incorporated llc ltd limited pvt private corp corporation co company plc pc pllc llp lp
the of and a sa sas sasu sarl eurl sci snc ei public""".split())
TOK = re.compile(r"[a-z0-9]+")
NUM = re.compile(r"\d+")
INDIC = re.compile(r"[\u0900-\u0dff]")


def fold(s):
    out = []
    for ch in unicodedata.normalize("NFKD", s.lower()):
        if unicodedata.combining(ch) and out and out[-1].isascii():
            continue
        out.append(ch)
    return "".join(out)


def core_tokens(name):
    return [t for t in TOK.findall(fold(name)) if t not in LEGAL and len(t) > 1]


def numbers(addr):
    return [n.lstrip("0") or "0" for n in NUM.findall(addr)]


def words(addr):
    return {t for t in TOK.findall(fold(addr)) if not t.isdigit() and len(t) > 2}


def pair_stats(rows):
    """rows: iterable of (s1_name, s1_addr, sx_name, sx_addr) -> dict of rates."""
    keys = ["name_overlap", "same_primary_num", "any_num", "addr_words>=2", "sx_addr_empty",
            "sx_name_indic", "name_or_num", "name_or_num_or_addr", "none_of_them", "name_jacc>=0.5"]
    cnt = dict.fromkeys(keys, 0)
    n = 0
    for a, b, c, d in rows:
        n += 1
        ta, tc = set(core_tokens(a)), set(core_tokens(c))
        inter = ta & tc
        nm = bool(inter)
        na, nc = numbers(b), set(numbers(d))
        prim = bool(na) and na[0] in nc
        anyn = bool(set(na) & nc)
        aw = len(words(b) & words(d)) >= 2
        cnt["name_overlap"] += nm
        cnt["same_primary_num"] += prim
        cnt["any_num"] += anyn
        cnt["addr_words>=2"] += aw
        cnt["sx_addr_empty"] += (d.strip() == "")
        cnt["sx_name_indic"] += bool(INDIC.search(c))
        cnt["name_or_num"] += (nm or prim)
        cnt["name_or_num_or_addr"] += (nm or prim or aw)
        cnt["none_of_them"] += not (nm or prim or aw)
        cnt["name_jacc>=0.5"] += (len(inter) / max(1, len(ta | tc)) >= 0.5)
    return n, {k: round(v / max(n, 1), 4) for k, v in cnt.items()}


s1_all = pl.read_parquet(f"{C}/train_s1.parquet")
pairs = pl.read_parquet(f"{C}/train_gt_pairs.parquet")
for country in ("US", "India"):
    s1 = s1_all.filter(pl.col("country") == country).sample(N_S1, seed=11)
    sx = pl.concat([pl.read_parquet(f"{C}/train_s{s}.parquet") for s in (2, 3)]).filter(pl.col("country") == country)
    gtp = pairs.join(s1.select(pl.col("entity_id").alias("s1")), on="s1")
    pos = (gtp.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"),
                              pl.col("business_address").alias("a1")), on="s1")
           .join(sx.select(pl.col("entity_id").alias("sx"), pl.col("business_name").alias("nx"),
                           pl.col("business_address").alias("ax")), on="sx"))
    n, st = pair_stats(pos.select("n1", "a1", "nx", "ax").iter_rows())
    print(f"\n===== {country}: TRUE pairs (n={n})", flush=True)
    for k, v in st.items():
        print(f"  {k:22s} {v}")
    by_src = pos.with_columns(pl.col("sx").str.slice(0, 2).alias("src"))
    for src in ("S2", "S3"):
        n2, st2 = pair_stats(by_src.filter(pl.col("src") == src).select("n1", "a1", "nx", "ax").iter_rows())
        print(f"  [{src}] name_overlap={st2['name_overlap']} same_num={st2['same_primary_num']} "
              f"addr_empty={st2['sx_addr_empty']} indic_name={st2['sx_name_indic']} none={st2['none_of_them']}")

    # Lookalike negatives: SX sharing the S1's rarest name token, not a true match.
    # Pure Python on purpose: polars list ops (map_elements/explode) crashed natively here.
    from collections import Counter, defaultdict
    import random

    sx_ids = sx["entity_id"].to_list()
    sx_names = sx["business_name"].to_list()
    sx_addrs = sx["business_address"].to_list()
    sx_toks = [set(core_tokens(nm)) for nm in sx_names]
    dfc = Counter(t for ts in sx_toks for t in ts)
    s1_rows = s1.select("entity_id", "business_name", "business_address").rows()
    rarest = {}
    for eid, nm, _ in s1_rows:
        cand = [t for t in set(core_tokens(nm)) if dfc.get(t, 0) >= 2]
        if cand:
            rarest[eid] = min(cand, key=dfc.__getitem__)
    need = set(rarest.values())
    postings = defaultdict(list)
    for i, ts in enumerate(sx_toks):
        for t in ts & need:
            postings[t].append(i)
    truth = set(zip(gtp["s1"].to_list(), gtp["sx"].to_list()))
    rnd = random.Random(0)
    neg_rows = []
    for eid, nm, ad in s1_rows:
        t = rarest.get(eid)
        if t is None:
            continue
        cands = [i for i in postings[t] if (eid, sx_ids[i]) not in truth]
        for i in rnd.sample(cands, min(3, len(cands))):
            neg_rows.append((nm, ad, sx_names[i], sx_addrs[i]))
    n, st = pair_stats(neg_rows)
    print(f"===== {country}: LOOKALIKE negatives sharing rarest name token (n={n})", flush=True)
    for k, v in st.items():
        print(f"  {k:22s} {v}")
    hard = [r for r in neg_rows
            if (lambda na, nc: bool(na) and na[0] in nc)(numbers(r[1]), set(numbers(r[3])))][:8]
    print("  examples of negatives that share name token AND house number:")
    for r in hard:
        print(f"    S1: {r[0]} | {r[1]}\n    SX: {r[2]} | {r[3]}")
    none = [r for r in pos.select("n1", "a1", "nx", "ax").iter_rows()
            if not set(core_tokens(r[0])) & set(core_tokens(r[2]))
            and not (lambda na, nc: bool(na) and na[0] in nc)(numbers(r[1]), set(numbers(r[3])))][:8]
    print("  examples of TRUE pairs with no name-token overlap and no house-number match:")
    for r in none:
        print(f"    S1: {r[0]} | {r[1]}\n    SX: {r[2]} | {r[3]}")
    del sx, sx_toks, dfc, postings, pos
