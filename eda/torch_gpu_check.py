"""Verify CUDA torch and time a blocking-style workload: chunked similarity matmul + top-k on GPU."""
import time

import torch

print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
p = torch.cuda.get_device_properties(0)
print(f"device: {p.name} | VRAM {p.total_memory / 2**30:.1f} GB | capability {p.major}.{p.minor}")

db = torch.nn.functional.normalize(torch.randn(500_000, 128, device="cuda"), dim=1)
q = torch.nn.functional.normalize(torch.randn(50_000, 128, device="cuda"), dim=1)


def run(dtype, chunk):
    d, qq = db.to(dtype), q.to(dtype)
    (qq[:chunk] @ d.T).topk(30, dim=1)  # warm-up
    torch.cuda.synchronize()
    t = time.time()
    for i in range(0, len(qq), chunk):
        (qq[i:i + chunk] @ d.T).topk(30, dim=1)
    torch.cuda.synchronize()
    dt = time.time() - t
    print(f"{str(dtype):14s} chunk={chunk}: 50k x 500k + top-30 in {dt:.2f}s (~{50_000 * 500_000 / dt / 1e9:.1f} G pairs/s)")


for dtype in (torch.float16, torch.float32):
    run(dtype, 1024)
