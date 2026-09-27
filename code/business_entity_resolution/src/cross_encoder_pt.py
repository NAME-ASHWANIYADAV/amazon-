"""Pretrained multilingual cross-encoder (MiniLM, Apache-2.0, 118M parameters) fine-tuned on grey-zone candidate
pairs. Reads "S1 name | S1 address" and "SX name | SX address" as one sequence, so attention compares the two
records token by token, with pretrained knowledge of French/English words that the hashed char n-grams and
the hand-made features do not have. Its logit is blended with the judge probability on the grey zone only."""
import math
import time

import numpy as np
import torch
from torch import nn

MODEL_ID = r"C:\amazon_ml\work\models\mminilm"   # local copy of cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (Apache-2.0)
MAX_LEN = int(__import__("os").environ.get("CE_MAX_LEN", "72"))


def load(model_id=MODEL_ID, device="cuda"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1, ignore_mismatched_sizes=True)
    return tok, model.to(device)


def texts(names, addrs):
    return [f"{n} | {a}" if a else n for n, a in zip(names, addrs)]


def _batches(tok, a, b, idx, batch, device):
    for s in range(0, len(idx), batch):
        i = idx[s:s + batch]
        enc = tok([a[k] for k in i], [b[k] for k in i], truncation=True, max_length=MAX_LEN, padding=True,
                  return_tensors="pt")
        yield {k: v.to(device) for k, v in enc.items()}, i


def finetune(tok, model, a, b, y, epochs=1, batch=64, lr=4e-5, log=print, seed=0, device="cuda"):
    """a/b: lists of pair texts; y: 0/1 labels."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(y) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.06,
                                                anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda")
    lossf = nn.BCEWithLogitsLoss()
    Y = torch.from_numpy(y.astype(np.float32))
    model.train()
    step, run, t0 = 0, 0.0, time.time()
    for ep in range(epochs):
        perm = np.random.default_rng(seed + ep).permutation(len(y))
        for enc, i in _batches(tok, a, b, perm, batch, device):
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(**enc).logits.squeeze(-1)
                loss = lossf(out.float(), Y[i].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
            if step % 200 == 0:
                log(f"  ce-pt epoch {ep} step {step}/{steps} loss {run:.4f} ({time.time() - t0:.0f}s)")
    model.eval()
    return model


@torch.inference_mode()
def score(tok, model, a, b, batch=512, device="cuda", tok_chunk=8192, log=None, budget=None):
    """Logits for the pairs in order (NaN for pairs not reached within `budget` seconds): tokenisation in large
    chunks, GPU batches. Callers put the pairs closest to the decision boundary first."""
    out = np.full(len(a), np.nan, dtype=np.float32)
    model.eval()
    t0 = time.time()
    for s in range(0, len(a), tok_chunk):
        if budget is not None and time.time() - t0 > budget:
            if log:
                log(f"  ce budget reached after {s}/{len(a)} pairs")
            break
        e = min(len(a), s + tok_chunk)
        enc = tok(a[s:e], b[s:e], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        for q in range(s, e, batch):
            sl = slice(q - s, min(e, q + batch) - s)
            part = {k: v[sl].to(device) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                out[q:q + (sl.stop - sl.start)] = model(**part).logits.squeeze(-1).float().cpu().numpy()
        if log and (s // tok_chunk) % 25 == 0:
            log(f"  ce scored {e}/{len(a)}")
    return out


def blend(p, ce, w):
    """Logistic blend of the judge probability and the cross-encoder logit: w = (w_logit, w_ce, bias)."""
    lz = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    z = w[0] * lz + w[1] * ce + w[2]
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
