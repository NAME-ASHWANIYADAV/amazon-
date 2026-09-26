"""Neural fingerprint bi-encoder: hashed n-gram bags -> unit name and address vectors."""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EMB_DIM, HASH_BUCKETS
from .featurize import hash_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FingerprintModel(nn.Module):
    """Shared hashed-n-gram table with a learned per-bucket weight, one small MLP tower per field."""

    def __init__(self, dim=EMB_DIM, buckets=2 * HASH_BUCKETS):
        super().__init__()
        self.emb = nn.EmbeddingBag(buckets, dim, mode="sum", sparse=True)
        self.gate = nn.Embedding(buckets, 1, sparse=True)
        nn.init.normal_(self.emb.weight, std=0.05)
        nn.init.constant_(self.gate.weight, math.log(math.e - 1))  # softplus(gate) starts at 1
        self.name_mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.addr_mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def _field(self, idx, off, bag, mlp):
        w = F.softplus(self.gate(idx)).squeeze(-1)
        v = self.emb(idx, off, per_sample_weights=w)
        wsum = torch.zeros(v.shape[0], device=v.device).index_add_(0, bag, w)
        return F.normalize(mlp(v / wsum.clamp_min(1e-6).unsqueeze(-1)), dim=-1)

    def forward(self, name_bags, addr_bags, addr_mask):
        n = self._field(*name_bags, self.name_mlp)
        a = self._field(*addr_bags, self.addr_mlp) * addr_mask.unsqueeze(-1)
        return n, a, F.normalize(torch.cat([n, a], dim=-1), dim=-1)


def _bags(strings, field_offset):
    idx, off = hash_features(strings, field_offset)
    lengths = np.diff(np.append(off, len(idx)))
    bag = np.repeat(np.arange(len(off)), lengths)
    return (torch.from_numpy(idx).to(DEVICE), torch.from_numpy(off).to(DEVICE),
            torch.from_numpy(bag).to(DEVICE))


def _forward(model, names, addrs):
    mask = torch.tensor([1.0 if a else 0.0 for a in addrs], device=DEVICE)
    return model(_bags(names, 0), _bags(addrs, HASH_BUCKETS), mask)


def _info_nce(q, k, tau):
    logits = q @ k.t() / tau
    y = torch.arange(q.shape[0], device=q.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.t(), y))


def train_encoder(s1_names, s1_addrs, sx_names, sx_addrs, pair_s1, pair_sx, epochs=8, batch=8192,
                  tau=0.05, seed=0, log=print):
    """Contrastive training. Each epoch uses one random SX per S1, so in-batch negatives never share an S1."""
    torch.manual_seed(seed)
    model = FingerprintModel().to(DEVICE)
    opt_s = torch.optim.SparseAdam([model.emb.weight, model.gate.weight], lr=0.01)
    opt_d = torch.optim.Adam([p for n, p in model.named_parameters() if not n.startswith(("emb.", "gate."))],
                             lr=2e-3)
    order = np.argsort(pair_s1, kind="stable")
    ps1, psx = pair_s1[order], pair_sx[order]
    starts = np.flatnonzero(np.r_[True, ps1[1:] != ps1[:-1]])
    counts = np.diff(np.r_[starts, len(ps1)])
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        pick = starts + (rng.random(len(starts)) * counts).astype(np.int64)
        pick = pick[rng.permutation(len(pick))]
        t0, tot, acc, nb = time.time(), 0.0, 0.0, 0
        model.train()
        for b in range(0, len(pick) - batch + 1, batch):
            sel = pick[b:b + batch]
            i1, ix = ps1[sel], psx[sel]
            n1, a1, r1 = _forward(model, s1_names.gather(i1).to_list(), s1_addrs.gather(i1).to_list())
            nx, ax, rx = _forward(model, sx_names.gather(ix).to_list(), sx_addrs.gather(ix).to_list())
            loss = _info_nce(r1, rx, tau) + 0.5 * _info_nce(n1, nx, tau)
            both = (a1.abs().sum(-1) > 0) & (ax.abs().sum(-1) > 0)
            if int(both.sum()) > 1:
                loss = loss + 0.5 * _info_nce(a1[both], ax[both], tau)
            opt_s.zero_grad()
            opt_d.zero_grad()
            loss.backward()
            opt_s.step()
            opt_d.step()
            with torch.no_grad():
                acc += ((r1 @ rx.t()).argmax(1) == torch.arange(len(sel), device=DEVICE)).float().mean().item()
            tot, nb = tot + loss.item(), nb + 1
        log(f"epoch {ep}: loss {tot / nb:.4f}  in-batch top1 {acc / nb:.4f}  {time.time() - t0:.0f}s")
    return model


@torch.no_grad()
def encode_to(model, names, addrs, path, batch=65536):
    """Write float16 record vectors for every row to a .npy memmap at `path`."""
    model.eval()
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(names), 2 * EMB_DIM))
    for b in range(0, len(names), batch):
        _, _, r = _forward(model, names.slice(b, batch).to_list(), addrs.slice(b, batch).to_list())
        out[b:b + batch] = r.half().cpu().numpy()
    out.flush()
    return out
