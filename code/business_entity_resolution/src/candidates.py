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
def knn(q_emb, d_emb, k, q_chunk=1024, d_block=300_000):
    D = torch.from_numpy(np.ascontiguousarray(d_emb)).to(DEVICE)  # float16 on GPU
    k = min(k, len(D))
    nq = len(q_emb)
    out = {n: np.empty((nq, k), dtype=np.float32) for n in ("cos", "cos_name", "cos_addr")}
    out_idx = np.empty((nq, k), dtype=np.int64)
    for s in range(0, nq, q_chunk):
        Q = torch.from_numpy(np.asarray(q_emb[s:s + q_chunk], dtype=np.float32)).to(DEVICE)
        best_v = torch.full((len(Q), k), -2.0, device=DEVICE)
        best_i = torch.zeros((len(Q), k), dtype=torch.int64, device=DEVICE)
        for b in range(0, len(D), d_block):
            sims = Q @ D[b:b + d_block].float().T
            v, i = sims.topk(min(k, sims.shape[1]), dim=1)
            v, i = torch.cat([best_v, v], 1), torch.cat([best_i, i + b], 1)
            best_v, pos = v.topk(k, dim=1)
            best_i = i.gather(1, pos)
        Dk = D[best_i].float()
        out["cos"][s:s + len(Q)] = best_v.cpu().numpy()
        out["cos_name"][s:s + len(Q)] = _half_cos(Q[:, :EMB_DIM], Dk[..., :EMB_DIM]).cpu().numpy()
        out["cos_addr"][s:s + len(Q)] = _half_cos(Q[:, EMB_DIM:], Dk[..., EMB_DIM:]).cpu().numpy()
        out_idx[s:s + len(Q)] = best_i.cpu().numpy()
    del D
    torch.cuda.empty_cache()
    return out_idx, out["cos"], out["cos_name"], out["cos_addr"]


def build_candidates(s1_country, sx_country, q_rows, s1_emb, sx_emb, k, log=print):
    """kNN per country for the S1 rows in q_rows. Returns a polars frame sorted by (s1, rank)."""
    frames = []
    for c in sorted(set(s1_country[q_rows].tolist())):
        qr = q_rows[s1_country[q_rows] == c]
        dr = np.flatnonzero(sx_country == c)
        log(f"knn {c}: {len(qr)} queries x {len(dr)} records")
        idx, cos, cn, ca = knn(s1_emb[qr], sx_emb[dr], k)
        kk = idx.shape[1]
        frames.append(pl.DataFrame({
            "s1": np.repeat(qr, kk).astype(np.int32), "sx": dr[idx.ravel()].astype(np.int32),
            "cos": cos.ravel(), "cos_name": cn.ravel(), "cos_addr": ca.ravel(),
            "rank": np.tile(np.arange(kk, dtype=np.int16), len(qr))}))
    return pl.concat(frames).sort("s1", "rank")


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
