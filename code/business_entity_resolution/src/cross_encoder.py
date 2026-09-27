"""Character-level cross-encoder: a small transformer reads "S1 name | S1 address [SEP] SX name | SX address" as
one sequence, so attention compares the two records character by character (complementary to the judge's
hand-made similarity features). Trained from scratch on labelled candidate pairs; its logit is blended with the
judge's probability."""
import math

import numpy as np
import torch
import torch.nn as nn

SIDE = 64                     # characters per record (name | address, truncated)
PAD, CLS, SEP, UNK = 0, 1, 2, 3
_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 |"
VOCAB = 4 + len(_CHARS)
_LUT = np.full(256, UNK, dtype=np.uint8)
for _i, _c in enumerate(_CHARS):
    _LUT[ord(_c)] = 4 + _i


def encode_records(names, addrs):
    """uint8 (n, SIDE) token matrix of 'name | address' (normalized, lowercase ascii text)."""
    out = np.zeros((len(names), SIDE), dtype=np.uint8)
    for i, (n, a) in enumerate(zip(names, addrs)):
        b = f"{n} | {a}".encode("ascii", "replace")[:SIDE]
        out[i, :len(b)] = _LUT[np.frombuffer(b, dtype=np.uint8)]
    return out


class PairEncoder(nn.Module):
    def __init__(self, d=128, heads=4, layers=3, ff=512, dropout=0.1):
        super().__init__()
        L = 2 * SIDE + 2
        self.tok = nn.Embedding(VOCAB, d, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
        self.seg = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        seg = torch.zeros(L, dtype=torch.long)
        seg[SIDE + 2:] = 1
        self.register_buffer("segids", seg, persistent=False)

    def forward(self, x):                                   # x: (B, 2*SIDE+2) long
        h = self.tok(x) + self.pos + self.seg(self.segids)
        h = self.enc(h, src_key_padding_mask=(x == PAD))
        return self.head(self.norm(h[:, 0])).squeeze(-1)


def build_batch(t1, tx, s1, sx):
    """Token tensors on device: t1 (n1, SIDE), tx (nx, SIDE) uint8; s1/sx long index tensors."""
    b = len(s1)
    cls = torch.full((b, 1), CLS, dtype=torch.long, device=t1.device)
    sep = torch.full((b, 1), SEP, dtype=torch.long, device=t1.device)
    return torch.cat([cls, t1[s1].long(), sep, tx[sx].long()], dim=1)


def train_ce(t1, tx, s1, sx, y, epochs=2, batch=512, lr=1e-3, log=print, seed=0):
    """t1/tx: uint8 numpy token matrices; s1/sx: int arrays indexing them; y: 0/1 labels."""
    torch.manual_seed(seed)
    dev = "cuda"
    T1, TX = torch.from_numpy(t1).to(dev), torch.from_numpy(tx).to(dev)
    S1, SX = torch.from_numpy(s1.astype(np.int64)), torch.from_numpy(sx.astype(np.int64))
    Y = torch.from_numpy(y.astype(np.float32))
    model = PairEncoder().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(y) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    scaler = torch.cuda.amp.GradScaler()
    lossf = nn.BCEWithLogitsLoss()
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(len(y))
        run = 0.0
        for b in range(0, len(y), batch):
            idx = perm[b:b + batch]
            x = build_batch(T1, TX, S1[idx].to(dev), SX[idx].to(dev))
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(x), Y[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 500 == 0:
                log(f"  ce epoch {ep} step {step}/{steps} loss {run:.4f}")
    del T1, TX
    torch.cuda.empty_cache()
    return model


@torch.no_grad()
def score_ce(model, t1, tx, s1, sx, batch=2048):
    dev = "cuda"
    model.eval()
    T1, TX = torch.from_numpy(t1).to(dev), torch.from_numpy(tx).to(dev)
    out = np.empty(len(s1), dtype=np.float32)
    for b in range(0, len(s1), batch):
        x = build_batch(T1, TX, torch.from_numpy(s1[b:b + batch].astype(np.int64)).to(dev),
                        torch.from_numpy(sx[b:b + batch].astype(np.int64)).to(dev))
        with torch.autocast("cuda", dtype=torch.float16):
            out[b:b + batch] = model(x).float().cpu().numpy()
    del T1, TX
    torch.cuda.empty_cache()
    return out                                               # logits
