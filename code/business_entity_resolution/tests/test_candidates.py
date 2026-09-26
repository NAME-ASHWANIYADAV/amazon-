import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _unit(v):
    return v / np.linalg.norm(v)


def _record(name, addr):
    n, a = _unit(name), (_unit(addr) if addr is not None else np.zeros_like(name))
    r = np.concatenate([n, a])
    return (r / np.linalg.norm(r)).astype(np.float16)


def test_address_pass_finds_same_address_with_unrelated_name():
    from src.candidates import build_candidates
    rng = np.random.default_rng(0)
    dim = 64
    q_name, q_addr = rng.normal(size=dim), rng.normal(size=dim)
    s1_emb = np.stack([_record(q_name, q_addr)])
    # 30 decoys share the name but not the address; the true record shares only the address
    sx = [_record(q_name + 0.05 * rng.normal(size=dim), rng.normal(size=dim)) for _ in range(30)]
    sx.append(_record(rng.normal(size=dim), q_addr))
    sx_emb = np.stack(sx)
    country = np.array(["US"], dtype=object)
    sx_country = np.array(["US"] * len(sx), dtype=object)
    only_record = build_candidates(country, sx_country, np.array([0]), s1_emb, sx_emb, k=5)
    with_addr = build_candidates(country, sx_country, np.array([0]), s1_emb, sx_emb, k=5, k_addr=1)
    assert 30 not in only_record["sx"].to_list()
    assert 30 in with_addr["sx"].to_list()
    assert with_addr["rank"].to_list() == list(range(with_addr.height))
    assert with_addr.select(["s1", "sx"]).is_unique().all()

    from src.candidates import add_address_pass
    merged = add_address_pass(only_record, country, sx_country, s1_emb, sx_emb, k_addr=1)
    assert sorted(merged["sx"].to_list()) == sorted(with_addr["sx"].to_list())
    assert merged["rank"].to_list() == list(range(merged.height))
