"""GPU brute-force cosine kNN between S1 and SX fingerprint vectors (per country)."""
import numpy as np
import polars as pl
import torch

from .config import EMB_DIM

DEVICE = "cuda"


def _half_cos(q, d):
    """Cosine between matching halves; -1 where either half is all-zero (empty address)."""
    num = (q.unsqueeze(1) * d).sum(-1)
    den = q.norm(dim=-1, keepdim=True) * d.norm(dim=-1)
    return torch.where(den > 1e-6, num / den.clamp_min(1e-6), torch.full_like(num, -1.0))


@torch.no_grad()
def knn(q_emb, d_emb, k, d_scale=None, by="record", q_chunk=1024, d_block=250_000):
    """Top-k SX per query, ranked by record cosine (by="record") or address-half cosine (by="addr").

    `d_scale` (per SX row) rescales records before the dot product: an empty-address SX has vector
    [name, 0], whose cosine with a full S1 record is capped at 1/sqrt(2); scaling it by sqrt(2) puts it
    on the same 0..1 scale as full records (cos = cos_name). The returned `cos` is always the record
    cosine; `cos_name` / `cos_addr` are the half cosines of the kept neighbours."""
    D = torch.from_numpy(np.ascontiguousarray(d_emb)).to(DEVICE)  # float16 on GPU
    if d_scale is not None:
        D *= torch.from_numpy(np.asarray(d_scale, dtype=np.float16)).to(DEVICE).unsqueeze(1)
    k = min(k, len(D))
    nq = len(q_emb)
    out = {n: np.empty((nq, k), dtype=np.float32) for n in ("cos", "cos_name", "cos_addr")}
    out_idx = np.empty((nq, k), dtype=np.int64)
    for s in range(0, nq, q_chunk):
        Q = torch.from_numpy(np.asarray(q_emb[s:s + q_chunk], dtype=np.float32)).to(DEVICE)
        Qr = Q if by == "record" else torch.nn.functional.normalize(Q[:, EMB_DIM:], dim=1)
        best_v = torch.full((len(Q), k), -2.0, device=DEVICE)
        best_i = torch.zeros((len(Q), k), dtype=torch.int64, device=DEVICE)
        for b in range(0, len(D), d_block):
            blk = D[b:b + d_block].float()
            if by == "addr":
                blk = torch.nn.functional.normalize(blk[:, EMB_DIM:], dim=1)
            sims = Qr @ blk.T
            del blk
            v, i = sims.topk(min(k, sims.shape[1]), dim=1)
            del sims  # free before the next block's matmul: keeps one sims buffer alive, not two
            v, i = torch.cat([best_v, v], 1), torch.cat([best_i, i + b], 1)
            best_v, pos = v.topk(k, dim=1)
            best_i = i.gather(1, pos)
            del v, i, pos
        Dk = D[best_i].float()
        out["cos"][s:s + len(Q)] = (Q.unsqueeze(1) * Dk).sum(-1).cpu().numpy()
        out["cos_name"][s:s + len(Q)] = _half_cos(Q[:, :EMB_DIM], Dk[..., :EMB_DIM]).cpu().numpy()
        out["cos_addr"][s:s + len(Q)] = _half_cos(Q[:, EMB_DIM:], Dk[..., EMB_DIM:]).cpu().numpy()
        out_idx[s:s + len(Q)] = best_i.cpu().numpy()
    del D
    torch.cuda.empty_cache()
    return out_idx, out["cos"], out["cos_name"], out["cos_addr"]


def _pairs_frame(qr, dr, idx, cos, cn, ca):
    kk = idx.shape[1]
    return pl.DataFrame({"s1": np.repeat(qr, kk).astype(np.int32), "sx": dr[idx.ravel()].astype(np.int32),
                         "cos": cos.ravel(), "cos_name": cn.ravel(), "cos_addr": ca.ravel()})


def rank_by_cos(cand):
    """Sort by (s1, cos desc) and number candidates 0.. within each S1 (numpy, no window functions)."""
    cand = cand.sort(["s1", "cos"], descending=[False, True])
    s1 = cand["s1"].to_numpy()
    starts = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]])
    counts = np.diff(np.r_[starts, len(s1)])
    rank = (np.arange(len(s1)) - np.repeat(starts, counts)).astype(np.int16)
    return cand.with_columns(pl.Series("rank", rank))


def build_candidates(s1_country, sx_country, q_rows, s1_emb, sx_emb, k, sx_addr_empty=None, k_addr=0,
                     log=print):
    """kNN per country for the S1 rows in q_rows: top-k by record cosine, plus (k_addr > 0) the top-k_addr
    by address cosine, which recovers matches whose name is an unrelated trade name. Returns a polars
    frame of unique (s1, sx) pairs sorted by (s1, rank), rank ordered by record cosine."""
    frames = []
    for c in sorted(set(s1_country[q_rows].tolist())):
        qr = q_rows[s1_country[q_rows] == c]
        dr = np.flatnonzero(sx_country == c)
        log(f"knn {c}: {len(qr)} queries x {len(dr)} records")
        scale = None if sx_addr_empty is None else np.where(sx_addr_empty[dr], np.sqrt(2.0), 1.0)
        q, d = s1_emb[qr], sx_emb[dr]
        frames.append(_pairs_frame(qr, dr, *knn(q, d, k, d_scale=scale)))
        if k_addr:
            log(f"knn {c}: address pass top-{k_addr}")
            frames.append(_pairs_frame(qr, dr, *knn(q, d, k_addr, d_scale=scale, by="addr")))
        del q, d
    cand = pl.concat(frames)
    if k_addr:
        cand = cand.unique(["s1", "sx"], keep="first")
    return rank_by_cos(cand)


def recall_report(cand, gt_rows, s1_rows, log=print):
    """Share of true (s1, sx) pairs of `s1_rows` found in `cand`, by K and by cosine floor."""
    truth = gt_rows.filter(pl.col("s1_row").is_in(pl.Series(np.asarray(s1_rows, dtype=np.int32)).implode()))
    hit = truth.join(cand.select(pl.col("s1").alias("s1_row"), pl.col("sx").alias("sx_row"), "rank", "cos"),
                     on=["s1_row", "sx_row"], how="left")
    n = truth.height
    for K in (5, 10, 20, 30, 40):
        log(f"  recall@{K}: {hit.filter(pl.col('rank') < K).height / n:.4f}")
    n_q = max(1, cand.filter(pl.col("s1").is_in(pl.Series(np.asarray(s1_rows, dtype=np.int32)).implode()))["s1"].n_unique())
    for f in (0.2, 0.3, 0.4, 0.5, 0.6):
        kept = cand.filter((pl.col("cos") >= f) & pl.col("s1").is_in(pl.Series(np.asarray(s1_rows, dtype=np.int32)).implode()))
        log(f"  floor {f}: recall {hit.filter(pl.col('cos') >= f).height / n:.4f}  pairs/S1 {kept.height / n_q:.1f}")
